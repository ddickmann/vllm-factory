# fused_ffn_t5.py
# SPDX-License-Identifier: Apache-2.0
# Elementwise-fused (GELU ∘ MUL ∘ Dropout) FFN for T5/mT5.
# Math-heavy GEMMs remain on cuBLASLt via vLLM Column/RowParallelLinear.

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Triton: FUSED GELU + MUL + (optional) DROPOUT  — elementwise only
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    # Tune once per "mode"; avoid re-tuning per N.
    key=['ACT_TYPE', 'use_dropout'],
)
@triton.jit
def _fused_gelu_mul_dropout_kernel(
    Gate_ptr,  # *flat* pointer
    Up_ptr,    # *flat* pointer
    Out_ptr,   # *flat* pointer
    dropout_p: tl.constexpr,
    dropout_seed: tl.constexpr,
    use_dropout: tl.constexpr,
    ACT_TYPE: tl.constexpr,  # 0: GELU(tanh), 1: GELU(exact), 2: ReLU, 3: SiLU
    N,                        # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < N

    # Load
    gate = tl.load(Gate_ptr + off, mask=mask, other=0.0)
    up   = tl.load(Up_ptr   + off, mask=mask, other=0.0)

    # Compute activation in fp32 for stability
    in_dtype = gate.dtype
    gate_f32 = gate.to(tl.float32)

    if ACT_TYPE == 0:  # GELU (tanh approximation)
        SQRT_2_OVER_PI = tl.full((), 0.7978845834732056, tl.float32)
        COEFF          = tl.full((), 0.044715,            tl.float32)
        x3     = gate_f32 * gate_f32 * gate_f32
        inner  = SQRT_2_OVER_PI * (gate_f32 + COEFF * x3)
        tanh_i = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        gate_f32 = 0.5 * gate_f32 * (1.0 + tanh_i)
    elif ACT_TYPE == 1:  # GELU (exact via erf)
        gate_f32 = 0.5 * gate_f32 * (1.0 + tl.erf(gate_f32 * 0.7071067811865476))
    elif ACT_TYPE == 2:  # ReLU
        gate_f32 = tl.maximum(gate_f32, 0.0)
    elif ACT_TYPE == 3:  # SiLU
        gate_f32 = gate_f32 * tl.sigmoid(gate_f32)

    # Elementwise mul (fp32) then cast back
    out_val = (gate_f32 * up.to(tl.float32)).to(in_dtype)

    # Optional dropout
    if use_dropout:
        # stateless rand: unique per program id, per element
        rnd = tl.rand(dropout_seed + pid, off)
        keep = rnd > dropout_p
        scale = 1.0 / (1.0 - dropout_p)
        out_val = tl.where(keep, out_val * scale, 0.0)

    # Store
    tl.store(Out_ptr + off, out_val, mask=mask)


def _act_type_map(act_fn: str) -> int:
    m = {
        'gelu': 0, 'gelu_new': 0, 'gelu_fast': 0, 'gelu_pytorch_tanh': 0,
        'gelu_accurate': 1,
        'relu': 2,
        'silu': 3, 'swish': 3,
    }
    return m.get(act_fn.lower(), 0)


def fused_gelu_mul_dropout(
    gate: torch.Tensor,
    up: torch.Tensor,
    act_fn: str = 'gelu',
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Fuse only the cheap elementwise ops; leave matmuls to cuBLASLt/vLLM.

    Args:
      gate, up: same shape tensors (bf16/fp16/fp32). Result dtype == input dtype.
      act_fn: 'gelu'|'gelu_accurate'|'relu'|'silu'/...
      dropout_p: only applied if training=True.
    """
    assert gate.shape == up.shape, "gate and up must have identical shapes"
    # Flatten as contiguous to guarantee 1D pointer arithmetic
    gate_flat = gate.contiguous().view(-1)
    up_flat   = up.contiguous().view(-1)
    N = gate_flat.numel()

    out = torch.empty_like(gate_flat)

    ACT_TYPE = _act_type_map(act_fn)
    use_dropout = bool(training and dropout_p > 0.0)
    # One random seed per call; deterministic across a single kernel launch
    dropout_seed = int(torch.randint(0, 2**31, (1,), device=gate_flat.device).item()) if use_dropout else 0

    # Grid: 1D launch over elements
    def grid(meta):
        return (triton.cdiv(N, meta['BLOCK_SIZE']),)

    _fused_gelu_mul_dropout_kernel[grid](
        gate_flat, up_flat, out,
        dropout_p=dropout_p,
        dropout_seed=dropout_seed,
        use_dropout=use_dropout,
        ACT_TYPE=ACT_TYPE,
        N=N,
    )
    return out.view_as(gate)


