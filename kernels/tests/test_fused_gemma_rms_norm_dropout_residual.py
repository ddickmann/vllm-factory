from __future__ import annotations

import torch

from kernels.fused_gemma_rms_norm_dropout_residual import (
    fused_gemma_rms_norm_dropout_residual,
)


def _reference_impl(
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    hidden_states = input_tensor.float()
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states * (1.0 + weight.float())
    hidden_states = hidden_states.to(input_tensor.dtype)
    return residual_tensor + hidden_states


def test_fused_gemma_rms_norm_dropout_residual_cpu_fallback() -> None:
    hidden_size = 64
    x = torch.randn(2, 3, hidden_size, dtype=torch.float32)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=torch.float32)

    actual = fused_gemma_rms_norm_dropout_residual(
        x,
        residual,
        weight,
        eps=1e-6,
        dropout_p=0.0,
        training=False,
    )
    expected = _reference_impl(x, residual, weight, eps=1e-6)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_fused_gemma_rms_norm_dropout_residual_cuda() -> None:
    if not torch.cuda.is_available():
        return

    hidden_size = 256
    x = torch.randn(3, 5, hidden_size, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)

    actual = fused_gemma_rms_norm_dropout_residual(
        x,
        residual,
        weight,
        eps=1e-6,
        dropout_p=0.0,
        training=False,
    )
    expected = _reference_impl(x, residual, weight, eps=1e-6)

    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)
