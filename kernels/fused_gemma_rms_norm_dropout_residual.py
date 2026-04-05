# SPDX-License-Identifier: Apache-2.0
"""Fused GemmaRMSNorm + dropout + residual add."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is expected on GPU runners.
    triton = None
    tl = None


def _reference_gemma_rms_norm_dropout_residual(
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    hidden_states = input_tensor.float()
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states * (1.0 + weight.float())
    hidden_states = hidden_states.to(input_tensor.dtype)
    hidden_states = F.dropout(hidden_states, p=dropout_p, training=training)
    return residual_tensor + hidden_states


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        ],
        key=["hidden_size", "use_dropout"],
    )
    @triton.jit
    def _fused_gemma_rms_norm_dropout_residual_kernel(
        X_ptr,
        Residual_ptr,
        Weight_ptr,
        Out_ptr,
        stride_x_row,
        stride_residual_row,
        stride_out_row,
        hidden_size,
        eps,
        dropout_p: tl.constexpr,
        dropout_seed: tl.constexpr,
        use_dropout: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)

        x_row_ptr = X_ptr + row_idx * stride_x_row
        residual_row_ptr = Residual_ptr + row_idx * stride_residual_row
        out_row_ptr = Out_ptr + row_idx * stride_out_row

        rms_acc = 0.0
        for block_start in range(0, hidden_size, BLOCK_SIZE):
            cols = block_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < hidden_size
            x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            rms_acc += tl.sum(x * x, axis=0)

        rms = tl.sqrt(rms_acc / hidden_size + eps)

        for block_start in range(0, hidden_size, BLOCK_SIZE):
            cols = block_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < hidden_size

            x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(
                residual_row_ptr + cols, mask=mask, other=0.0
            ).to(tl.float32)
            weight = tl.load(Weight_ptr + cols, mask=mask, other=0.0).to(
                tl.float32
            )

            hidden_states = x / rms
            hidden_states = hidden_states * (1.0 + weight)

            if use_dropout:
                offsets = row_idx * hidden_size + cols
                random_values = tl.rand(dropout_seed, offsets)
                keep = random_values > dropout_p
                scale = 1.0 / (1.0 - dropout_p)
                hidden_states = tl.where(keep, hidden_states * scale, 0.0)

            out = residual + hidden_states
            tl.store(out_row_ptr + cols, out, mask=mask)


def fused_gemma_rms_norm_dropout_residual(
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Compute `residual + dropout(GemmaRMSNorm(input))` in one pass."""

    if input_tensor.shape != residual_tensor.shape:
        raise ValueError("input_tensor and residual_tensor must have identical shapes.")
    if input_tensor.shape[-1] != weight.shape[0]:
        raise ValueError("The last dimension must match the GemmaRMSNorm weight.")

    if (
        triton is None
        or (not input_tensor.is_cuda)
        or (not residual_tensor.is_cuda)
        or (not weight.is_cuda)
    ):
        return _reference_gemma_rms_norm_dropout_residual(
            input_tensor,
            residual_tensor,
            weight,
            eps=eps,
            dropout_p=dropout_p,
            training=training,
        )

    input_shape = input_tensor.shape
    hidden_size = input_shape[-1]
    x_2d = input_tensor.view(-1, hidden_size).contiguous()
    residual_2d = residual_tensor.view(-1, hidden_size).contiguous()
    out_2d = torch.empty_like(x_2d)
    use_dropout = bool(training and dropout_p > 0.0)
    dropout_seed = (
        int(torch.randint(0, 2**31, (1,), device=x_2d.device).item())
        if use_dropout
        else 0
    )

    grid = (x_2d.shape[0],)
    _fused_gemma_rms_norm_dropout_residual_kernel[grid](
        x_2d,
        residual_2d,
        weight.contiguous(),
        out_2d,
        x_2d.stride(0),
        residual_2d.stride(0),
        out_2d.stride(0),
        hidden_size,
        eps,
        dropout_p=dropout_p,
        dropout_seed=dropout_seed,
        use_dropout=use_dropout,
    )
    return out_2d.view(input_shape)


__all__ = [
    "fused_gemma_rms_norm_dropout_residual",
]
