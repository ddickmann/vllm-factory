import math
from typing import Optional

import torch
import triton
import triton.language as tl

# ==============================================================================
# CORRECT RPB BUCKETING (from fused_t5_attention.py)
# ==============================================================================

@triton.jit
def _get_rp_bucket(
    relative_position,
    IS_BIDIRECTIONAL: tl.constexpr,
    CALC_NUM_BUCKETS: tl.constexpr,
    CALC_MAX_EXACT: tl.constexpr,
    LOG_MAX_DIST_RATIO: tl.constexpr,
):
    """
    CORRECT implementation of T5's bucketing logic.
    """
    relative_buckets = tl.zeros_like(relative_position)

    if IS_BIDIRECTIONAL:
        relative_buckets += (relative_position > 0).to(tl.int32) * CALC_NUM_BUCKETS
        relative_position = tl.abs(relative_position)
    else:
        relative_position = -tl.minimum(relative_position, 0)

    is_small = relative_position < CALC_MAX_EXACT

    relative_position_float = relative_position.to(tl.float32)

    if CALC_MAX_EXACT == 0 or LOG_MAX_DIST_RATIO == 0.0:
        rel_pos_if_large = tl.zeros_like(relative_position) + (CALC_NUM_BUCKETS - 1)
    else:
        ratio = tl.maximum(1.0, relative_position_float / CALC_MAX_EXACT)
        log_ratio = tl.log(ratio)

        rel_pos_if_large_float = CALC_MAX_EXACT + (
            log_ratio / LOG_MAX_DIST_RATIO * (CALC_NUM_BUCKETS - CALC_MAX_EXACT)
        )
        rel_pos_if_large = rel_pos_if_large_float.to(tl.int32)

    rel_pos_if_large = tl.minimum(rel_pos_if_large, CALC_NUM_BUCKETS - 1)

    bucket_indices = tl.where(
        is_small,
        relative_position,
        rel_pos_if_large
    )

    return bucket_indices + relative_buckets


# ==============================================================================
# OPTIMIZED TRITON KERNEL WITH SRAM GATHER
# ==============================================================================

@triton.jit
def _flash_attn_rpb_fwd_kernel_final(
    # Input/Output tensors
    Q, K, V, Out,
    RPB_Table,
    Padding_Mask,
    # Strides
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_rpb_bucket, stride_rpb_head,
    stride_mask_b, stride_mask_h, stride_mask_m, stride_mask_n,
    # Dimensions
    batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
    # RPB configuration
    IS_BIDIRECTIONAL: tl.constexpr,
    CALC_NUM_BUCKETS: tl.constexpr,
    CALC_MAX_EXACT: tl.constexpr,
    LOG_MAX_DIST_RATIO: tl.constexpr,
    TOTAL_NUM_BUCKETS: tl.constexpr,
    # Attention configuration
    IS_CAUSAL: tl.constexpr,
    USE_PADDING_MASK: tl.constexpr,
    USE_RPB: tl.constexpr,
    DROPOUT_P: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    # Numerical stability
    SOFTMAX_SCALE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # ==========================================================================
    # RPB TABLE ACCESS STRATEGY
    # ==========================================================================
    if USE_RPB:
        rpb_head_offset = pid_h * stride_rpb_head

    # ==========================================================================
    # Standard Flash Attention Setup
    # ==========================================================================
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (pid_b * stride_qb + pid_h * stride_qh +
                  offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q_mask = offs_m[:, None] < seq_len_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    if IS_CAUSAL:
        kv_end = tl.minimum((pid_m + 1) * BLOCK_M, seq_len_k)
    else:
        kv_end = seq_len_k

    num_blocks_n = tl.cdiv(kv_end, BLOCK_N)

    for block_n_idx in range(num_blocks_n):
        start_n = block_n_idx * BLOCK_N
        offs_n_curr = start_n + offs_n

        k_ptrs = K + (pid_b * stride_kb + pid_h * stride_kh +
                      offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        v_ptrs = V + (pid_b * stride_vb + pid_h * stride_vh +
                      offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd)

        kv_mask_n = offs_n_curr < seq_len_k
        k = tl.load(k_ptrs, mask=kv_mask_n[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask_n[:, None], other=0.0)

        qk = tl.dot(q, k)
        qk = qk * SOFTMAX_SCALE

        # Define positions *unconditionally* if (USE_RPB or IS_CAUSAL)
        if USE_RPB or IS_CAUSAL:
            q_positions = offs_m[:, None]
            k_positions = offs_n_curr[None, :]

        # ======================================================================
        # RPB COMPUTATION (NOW OPTIONAL)
        # ======================================================================
        if USE_RPB:
            relative_pos = k_positions - q_positions

            bucket_indices = _get_rp_bucket(
                relative_pos,
                IS_BIDIRECTIONAL=IS_BIDIRECTIONAL,
                CALC_NUM_BUCKETS=CALC_NUM_BUCKETS,
                CALC_MAX_EXACT=CALC_MAX_EXACT,
                LOG_MAX_DIST_RATIO=LOG_MAX_DIST_RATIO,
            )

            bucket_indices = tl.maximum(0, tl.minimum(bucket_indices, TOTAL_NUM_BUCKETS - 1))

            rpb_ptrs = RPB_Table + (
                bucket_indices * stride_rpb_bucket +
                rpb_head_offset
            )
            qk_mask_rpb = (offs_m[:, None] < seq_len_q) & (offs_n_curr[None, :] < seq_len_k)
            rpb_bias = tl.load(rpb_ptrs, mask=qk_mask_rpb, other=0.0)

            qk = qk + rpb_bias.to(qk.dtype)

        # ======================================================================
        # Apply masks
        # ======================================================================
        if IS_CAUSAL:
            causal_mask = k_positions > q_positions
            qk = tl.where(causal_mask, float("-inf"), qk)

        if USE_PADDING_MASK:
            mask_ptrs = Padding_Mask + (
                pid_b * stride_mask_b +
                pid_h * stride_mask_h +
                offs_m[:, None] * stride_mask_m +
                offs_n_curr[None, :] * stride_mask_n
            )
            qk_mask_padding = q_mask & kv_mask_n[None, :]
            padding_vals = tl.load(mask_ptrs, mask=qk_mask_padding, other=0.0)
            qk = qk + padding_vals.to(qk.dtype)

        qk_mask_final = (offs_m[:, None] < seq_len_q) & (offs_n_curr[None, :] < seq_len_k)
        qk = tl.where(qk_mask_final, qk, float("-inf"))

        # ======================================================================
        # Online Softmax
        # ======================================================================
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = m_new

    acc = acc / l_i[:, None]

    # ==========================================================================
    # Fused dropout (optional)
    # ==========================================================================
    if DROPOUT_P > 0.0:
        # --- THIS IS THE tl.rand FIX ---
        seed = pid_b * 10000 + pid_h * 1000 + pid_m
        dropout_offsets = offs_m[:, None]  # Use the tensor itself as the offset
        dropout_mask = tl.rand(seed, dropout_offsets) > DROPOUT_P
        acc = tl.where(dropout_mask, acc / (1.0 - DROPOUT_P), 0.0)

    # ==========================================================================
    # Write output
    # ==========================================================================
    out_ptrs = Out + (
        pid_b * stride_ob +
        pid_h * stride_oh +
        offs_m[:, None] * stride_om +
        offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


# ==============================================================================
# AUTO-TUNING
# ==============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
    ],
    key=['head_dim', 'IS_CAUSAL', 'USE_PADDING_MASK', 'USE_RPB', 'IS_BIDIRECTIONAL'],

)
@triton.jit
def _flash_attn_rpb_fwd_kernel_autotuned(
    Q, K, V, Out, RPB_Table, Padding_Mask,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_rpb_bucket, stride_rpb_head,
    stride_mask_b, stride_mask_h, stride_mask_m, stride_mask_n,
    batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
    IS_BIDIRECTIONAL: tl.constexpr,
    CALC_NUM_BUCKETS: tl.constexpr,
    CALC_MAX_EXACT: tl.constexpr,
    LOG_MAX_DIST_RATIO: tl.constexpr,
    TOTAL_NUM_BUCKETS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_PADDING_MASK: tl.constexpr,
    USE_RPB: tl.constexpr,
    DROPOUT_P: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
):
    """Autotuned version."""
    return _flash_attn_rpb_fwd_kernel_final(
        Q, K, V, Out, RPB_Table, Padding_Mask,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_rpb_bucket, stride_rpb_head,
        stride_mask_b, stride_mask_h, stride_mask_m, stride_mask_n,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
        IS_BIDIRECTIONAL, CALC_NUM_BUCKETS, CALC_MAX_EXACT,
        LOG_MAX_DIST_RATIO, TOTAL_NUM_BUCKETS,
        IS_CAUSAL, USE_PADDING_MASK,
        USE_RPB,
        DROPOUT_P,
        BLOCK_M, BLOCK_N, BLOCK_DMODEL, SOFTMAX_SCALE,
    )


# ==============================================================================
# PYTHON INTERFACE
# ==============================================================================

def flash_attention_rpb_final(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rpb_table: Optional[torch.Tensor],
    num_buckets: int,
    max_distance: int,
    is_decoder: bool,
    padding_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: float = 1.0,
    use_autotune: bool = True,
) -> torch.Tensor:

    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape

    out = torch.empty_like(q)

    IS_BIDIRECTIONAL = not is_decoder
    IS_CAUSAL = is_decoder

    if IS_BIDIRECTIONAL:
        CALC_NUM_BUCKETS = num_buckets // 2
    else:
        CALC_NUM_BUCKETS = num_buckets

    CALC_MAX_EXACT = CALC_NUM_BUCKETS // 2

    if CALC_MAX_EXACT == 0:
         ratio = 1.0
    else:
         ratio = max_distance / CALC_MAX_EXACT

    if ratio <= 1.0:
        LOG_MAX_DIST_RATIO = 0.0
    else:
        LOG_MAX_DIST_RATIO = math.log(ratio)

    # --- CRASH FIX + REVERT TO STANDARD LAYOUT ---
    USE_RPB = rpb_table is not None
    if USE_RPB:
        # We assume [Buckets, Heads] - The standard HF layout
        TOTAL_NUM_BUCKETS = rpb_table.shape[0]
        stride_rpb_bucket, stride_rpb_head = rpb_table.stride()
    else:
        TOTAL_NUM_BUCKETS = num_buckets
        # Create a valid 2D dummy to prevent stride unpacking crashes
        rpb_table = torch.empty((1, 1), device=q.device, dtype=q.dtype)
        stride_rpb_bucket, stride_rpb_head = 0, 0
    # ---------------------------------------------

    # Padding mask setup
    USE_PADDING_MASK = padding_mask is not None

    if not USE_PADDING_MASK:
        padding_mask = torch.empty(0, device=q.device, dtype=q.dtype)
        stride_mask_b = stride_mask_h = stride_mask_m = stride_mask_n = 0
    else:
        mask_strides = padding_mask.stride()
        mask_shape = padding_mask.shape
        stride_mask_b = mask_strides[0] if mask_shape[0] > 1 else 0
        stride_mask_h = mask_strides[1] if mask_shape[1] > 1 else 0
        stride_mask_m = mask_strides[2] if mask_shape[2] > 1 else 0
        stride_mask_n = mask_strides[3] if mask_shape[3] > 1 else 0

    # Import your kernels here or assume they are defined above
    # from .flash_attention_rpb import _flash_attn_rpb_fwd_kernel_autotuned, _flash_attn_rpb_fwd_kernel_final

    if use_autotune:
        def grid(META):
            return (
                    triton.cdiv(seq_len_q, META['BLOCK_M']),
                    batch_size,
                    num_heads,
                )

        _flash_attn_rpb_fwd_kernel_autotuned[grid](
            q, k, v, out, rpb_table, padding_mask,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            stride_rpb_bucket, stride_rpb_head,
            stride_mask_b, stride_mask_h, stride_mask_m, stride_mask_n,
            batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
            IS_BIDIRECTIONAL, CALC_NUM_BUCKETS, CALC_MAX_EXACT,
            LOG_MAX_DIST_RATIO, TOTAL_NUM_BUCKETS,
            IS_CAUSAL, USE_PADDING_MASK,
            USE_RPB,
            dropout_p,
            BLOCK_DMODEL=head_dim,
            SOFTMAX_SCALE=softmax_scale,
        )
    else:
        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(seq_len_q, BLOCK_M), batch_size, num_heads)

        _flash_attn_rpb_fwd_kernel_final[grid](
            q, k, v, out, rpb_table, padding_mask,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            stride_rpb_bucket, stride_rpb_head,
            stride_mask_b, stride_mask_h, stride_mask_m, stride_mask_n,
            batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
            IS_BIDIRECTIONAL, CALC_NUM_BUCKETS, CALC_MAX_EXACT,
            LOG_MAX_DIST_RATIO, TOTAL_NUM_BUCKETS,
            IS_CAUSAL, USE_PADDING_MASK,
            USE_RPB,
            dropout_p,
            BLOCK_M, BLOCK_N, head_dim, softmax_scale,
        )

    return out
