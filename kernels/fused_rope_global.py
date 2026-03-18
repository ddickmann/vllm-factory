# SPDX-License-Identifier: Apache-2.0
# Fused RoPE Application for ModernBERT (Global layers)
# Fuses rotate_half + cos/sin multiplication into single kernel

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8),
    ],
    key=['head_dim'],  # Tune per head_dim
)
@triton.jit
def _fused_rope_kernel(
    X_ptr,              # Input tensor [batch * seq_len * num_heads, head_dim]
    Cos_ptr,            # Cos values [batch, seq_len, head_dim]
    Sin_ptr,            # Sin values [batch, seq_len, head_dim]
    Out_ptr,            # Output tensor
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    stride_x_bsh,       # Stride for batch*seq*head dimension
    stride_x_d,         # Stride for head_dim
    stride_cos_batch,
    stride_cos_seq,
    stride_cos_dim,
    stride_sin_batch,
    stride_sin_seq,
    stride_sin_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused RoPE application kernel.

    Computes: out = x * cos + rotate_half(x) * sin
    where rotate_half(x) = [-x[..., d//2:], x[..., :d//2]]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate which batch, sequence, head this program handles
    bsh_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    dim_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Calculate batch, seq, head indices
    total_elements = batch_size * seq_len * num_heads
    mask_bsh = bsh_idx < total_elements
    mask_dim = dim_idx < head_dim

    # Load input chunk
    x_ptrs = X_ptr + bsh_idx[:, None] * stride_x_bsh + dim_idx[None, :] * stride_x_d
    x = tl.load(x_ptrs, mask=mask_bsh[:, None] & mask_dim[None, :], other=0.0)

    # Calculate batch and seq indices for cos/sin lookup
    batch_idx = bsh_idx // (seq_len * num_heads)
    seq_idx = (bsh_idx // num_heads) % seq_len

    # Load cos and sin values
    cos_ptrs = (Cos_ptr +
                batch_idx[:, None] * stride_cos_batch +
                seq_idx[:, None] * stride_cos_seq +
                dim_idx[None, :] * stride_cos_dim)
    sin_ptrs = (Sin_ptr +
                batch_idx[:, None] * stride_sin_batch +
                seq_idx[:, None] * stride_sin_seq +
                dim_idx[None, :] * stride_sin_dim)

    cos = tl.load(cos_ptrs, mask=mask_bsh[:, None] & mask_dim[None, :], other=1.0)
    sin = tl.load(sin_ptrs, mask=mask_bsh[:, None] & mask_dim[None, :], other=0.0)

    # Apply RoPE rotation
    # For each element at position d, we need:
    # - if d < head_dim//2: out[d] = x[d] * cos[d] - x[d + head_dim//2] * sin[d]
    # - if d >= head_dim//2: out[d] = x[d] * cos[d] + x[d - head_dim//2] * sin[d]

    half_dim = head_dim // 2
    is_first_half = dim_idx < half_dim

    # For first half: need to load x from second half (d + half_dim)
    # For second half: need to load x from first half (d - half_dim)
    pair_dim_idx = tl.where(is_first_half, dim_idx + half_dim, dim_idx - half_dim)
    pair_mask = pair_dim_idx < head_dim

    x_pair_ptrs = X_ptr + bsh_idx[:, None] * stride_x_bsh + pair_dim_idx[None, :] * stride_x_d
    x_pair = tl.load(x_pair_ptrs, mask=mask_bsh[:, None] & pair_mask[None, :], other=0.0)

    # Compute rotated output
    # rotate_half logic: [-x[d+half], x[d-half]] which means:
    # - first half gets: -x[d+half]
    # - second half gets: +x[d-half]
    rotated = tl.where(is_first_half[None, :], -x_pair, x_pair)

    # Final RoPE: x * cos + rotated * sin
    out = x * cos + rotated * sin

    # Store result
    out_ptrs = Out_ptr + bsh_idx[:, None] * stride_x_bsh + dim_idx[None, :] * stride_x_d
    tl.store(out_ptrs, out, mask=mask_bsh[:, None] & mask_dim[None, :])


def fused_rope_global_apply(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE (global) to query and key tensors using fused Triton kernel.

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

    # Apply RoPE to Q
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

