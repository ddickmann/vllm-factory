# SPDX-License-Identifier: Apache-2.0
# Fused RoPE Application for ModernBERT (Local layers)
# Uses same kernel as global, but with different theta-computed cos/sin tables

import torch
import triton

from .fused_rope_global import _fused_rope_kernel


def fused_rope_local_apply(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE (local) to query and key tensors using fused Triton kernel.

    This uses the same kernel as global RoPE, but receives cos/sin computed
    with the local layer's theta parameter.

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_heads, head_dim]
        cos: Cosine values [batch, seq_len, head_dim] or [batch, 1, seq_len, head_dim]
        sin: Sine values [batch, seq_len, head_dim] or [batch, 1, seq_len, head_dim]

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Handle broadcasting for cos/sin
    if cos.dim() == 4:  # [batch, 1, seq_len, head_dim]
        cos = cos.squeeze(1)
    if sin.dim() == 4:
        sin = sin.squeeze(1)

    # Flatten to [batch * seq_len * num_heads, head_dim] for kernel
    q_flat = q.reshape(-1, head_dim).contiguous()
    k_flat = k.reshape(-1, head_dim).contiguous()

    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)

    # Grid dimensions
    def grid(meta):
        return (
            triton.cdiv(batch_size * seq_len * num_heads, meta['BLOCK_SIZE_M']),
            triton.cdiv(head_dim, meta['BLOCK_SIZE_N']),
        )

    # Apply RoPE to Q (uses same kernel as global)
    _fused_rope_kernel[grid](
        q_flat, cos, sin, q_out,
        batch_size, seq_len, num_heads, head_dim,
        q_flat.stride(0), q_flat.stride(1),
        cos.stride(0), cos.stride(1), cos.stride(2),
        sin.stride(0), sin.stride(1), sin.stride(2),
    )

    # Apply RoPE to K
    _fused_rope_kernel[grid](
        k_flat, cos, sin, k_out,
        batch_size, seq_len, num_heads, head_dim,
        k_flat.stride(0), k_flat.stride(1),
        cos.stride(0), cos.stride(1), cos.stride(2),
        sin.stride(0), sin.stride(1), sin.stride(2),
    )

    # Reshape back to original shape
    q_rotated = q_out.view(batch_size, seq_len, num_heads, head_dim)
    k_rotated = k_out.view(batch_size, seq_len, num_heads, head_dim)

    return q_rotated, k_rotated

