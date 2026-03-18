# SPDX-License-Identifier: Apache-2.0
# Fused Dropout + Residual for ModernBERT
# Fuses dropout and residual add into single kernel

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['use_dropout'],  # Tune once per dropout mode
)
@triton.jit
def _fused_dropout_residual_kernel(
    Input_ptr,       # Input tensor (e.g., after Wo projection)
    Residual_ptr,    # Residual tensor to add
    Out_ptr,         # Output tensor
    dropout_p: tl.constexpr,
    dropout_seed: tl.constexpr,
    use_dropout: tl.constexpr,
    N,               # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused dropout + residual add kernel.

    Computes: out = residual + dropout(input)
    """
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < N

    # Load input and residual
    input_val = tl.load(Input_ptr + off, mask=mask, other=0.0)
    residual_val = tl.load(Residual_ptr + off, mask=mask, other=0.0)

    # Apply dropout if enabled
    if use_dropout:
        # Stateless random generation
        rnd = tl.rand(dropout_seed + pid, off)
        keep = rnd > dropout_p
        scale = 1.0 / (1.0 - dropout_p)
        input_val = tl.where(keep, input_val * scale, 0.0)

    # Add residual
    out_val = residual_val + input_val

    # Store result
    tl.store(Out_ptr + off, out_val, mask=mask)


def fused_dropout_residual(
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Fuse dropout and residual add into single kernel.

    This eliminates one memory read/write by fusing:
    - Dropout application
    - Residual addition

    Args:
        input_tensor: Input tensor (e.g., after linear projection)
        residual_tensor: Residual tensor to add
        dropout_p: Dropout probability (only applied if training=True)
        training: Whether in training mode

    Returns:
        Output tensor: residual + dropout(input)
    """
    assert input_tensor.shape == residual_tensor.shape, \
        "input and residual must have identical shapes"

    # Flatten for kernel
    input_flat = input_tensor.contiguous().view(-1)
    residual_flat = residual_tensor.contiguous().view(-1)
    N = input_flat.numel()

    out = torch.empty_like(input_flat)

    use_dropout = bool(training and dropout_p > 0.0)
    dropout_seed = int(torch.randint(0, 2**31, (1,), device=input_flat.device).item()) if use_dropout else 0

    # Grid: 1D launch over elements
    def grid(meta):
        return (triton.cdiv(N, meta['BLOCK_SIZE']),)

    _fused_dropout_residual_kernel[grid](
        input_flat, residual_flat, out,
        dropout_p=dropout_p,
        dropout_seed=dropout_seed,
        use_dropout=use_dropout,
        N=N,
    )

    return out.view_as(input_tensor)

