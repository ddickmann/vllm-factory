# SPDX-License-Identifier: Apache-2.0
"""
Flash Attention with DeBERTa Disentangled Relative Position Bias.

Ported from FlashDeBERTa (https://github.com/Knowledgator/FlashDeBERTa).
Forward-only (inference) kernel that fuses:
  1. Log-bucket relative position computation
  2. Content-to-Position (c2p) bias gather
  3. Position-to-Content (p2c) bias gather
  4. Flash Attention with online softmax

Input layout: Q/K/V (B, H, M, D)
  pos_key:   (B, H, M, 2*ATT_SPAN) — pre-computed Q @ pos_key_proj(rel_emb)^T
  pos_query: (B, H, N, 2*ATT_SPAN) — pre-computed K @ pos_query_proj(rel_emb)^T

Falls back to PyTorch SDPA when Triton is unavailable.
"""

import functools
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ==============================================================================
# Shared Memory Estimation
# ==============================================================================

def _calculate_shared_memory(BLOCK_M, BLOCK_N, BLOCK_DMODEL, num_stages, dtype,
                              has_c2p=False, has_p2c=False, ATT_SPAN=0):
    """Estimate shared memory usage for the fwd kernel."""
    dtype_size = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    q_size = BLOCK_M * BLOCK_DMODEL * dtype_size
    k_size = BLOCK_N * BLOCK_DMODEL * dtype_size
    v_size = BLOCK_N * BLOCK_DMODEL * dtype_size
    attn_size = BLOCK_M * BLOCK_N * dtype_size
    accum_size = BLOCK_M * BLOCK_DMODEL * dtype_size

    pos_mem = 0
    if has_c2p:
        pos_mem += BLOCK_M * 2 * ATT_SPAN * dtype_size
    if has_p2c:
        pos_mem += BLOCK_N * 2 * ATT_SPAN * dtype_size

    buffers = BLOCK_M * BLOCK_N * 4
    per_stage = q_size + k_size + v_size + attn_size + pos_mem + buffers
    return (num_stages * per_stage + accum_size) // 2


# ==============================================================================
# Config Cache
# ==============================================================================

def _cdiv(a, b):
    return (a + b - 1) // b


@functools.lru_cache(maxsize=128)
def _get_fwd_config(M, N, D, has_disentangled, att_span):
    """Auto-tune kernel block sizes based on GPU capability."""
    # Environment variable overrides
    env_keys = ['FLASHDEBERTA_FWD_BLOCK_M', 'FLASHDEBERTA_FWD_BLOCK_N',
                'FLASHDEBERTA_FWD_NUM_STAGES', 'FLASHDEBERTA_FWD_NUM_WARPS']
    if all(k in os.environ for k in env_keys):
        return tuple(int(os.environ[k]) for k in env_keys)

    cap = torch.cuda.get_device_capability()

    # Capability-based shared memory limits
    smem_map = {
        (7, 0): 96000, (7, 2): 96000, (7, 5): 64000,
        (8, 0): 163000, (8, 6): 99000, (8, 7): 163000, (8, 9): 99000,
        (9, 0): 227000,
    }
    prop = torch.cuda.get_device_properties(0)
    if hasattr(prop, "shared_memory_per_block_optin"):
        max_smem = prop.shared_memory_per_block_optin - 2000
    elif cap in smem_map:
        max_smem = smem_map[cap] - 2000
    elif cap[0] >= 8:
        max_smem = 97000
    else:
        max_smem = 46000

    # Default configs by capability — conservative for disentangled attention
    # (position bias tensors consume extra shared memory)
    if cap[0] >= 9:
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 2, 4
    elif cap[0] >= 8:
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 16, 16, 2, 4

    ATT_SPAN = att_span if has_disentangled else 0

    # Fit within shared memory
    smem = _calculate_shared_memory(BLOCK_M, BLOCK_N, D, num_stages, torch.float16,
                                     has_c2p=has_disentangled, has_p2c=has_disentangled,
                                     ATT_SPAN=ATT_SPAN)
    while smem > max_smem and (BLOCK_M > 16 or BLOCK_N > 16 or num_stages > 1):
        if num_stages > 1:
            num_stages -= 1
        elif BLOCK_M > 32 and BLOCK_N > 32:
            BLOCK_M //= 2
            BLOCK_N //= 2
        elif BLOCK_M > 32:
            BLOCK_M //= 2
        elif BLOCK_N > 32:
            BLOCK_N //= 2
        elif BLOCK_M > 16:
            BLOCK_M //= 2
        elif BLOCK_N > 16:
            BLOCK_N //= 2
        smem = _calculate_shared_memory(BLOCK_M, BLOCK_N, D, num_stages, torch.float16,
                                         has_c2p=has_disentangled, has_p2c=has_disentangled,
                                         ATT_SPAN=ATT_SPAN)

    return BLOCK_M, BLOCK_N, num_stages, num_warps


# ==============================================================================
# Triton Forward Kernel
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _flash_deberta_fwd_kernel(
        Q, K, V,
        K_POS, Q_POS,
        L, O,  # noqa: E741
        SEQ_LENGTHS,
        sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        stride_pk0, stride_pk1, stride_pk2, stride_pk3,
        stride_pq0, stride_pq1, stride_pq2, stride_pq3,
        Z, H, M, N, P_SEQ,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr, LARGER_M: tl.constexpr,
        DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
        HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
        ATT_SPAN: tl.constexpr,
        NUM_BUCKETS: tl.constexpr, MAX_DISTANCE: tl.constexpr,
        USE_LOG_BUCKET: tl.constexpr,
    ):
        input_dtype = Q.dtype.element_ty

        start_m = tl.program_id(0)
        off_h = tl.program_id(1)
        off_z = tl.program_id(2)

        log2e: tl.constexpr = 1.4426950408889634

        Q += off_z * stride_qz + off_h * stride_qh
        K += off_z * stride_kz + off_h * stride_kh
        V += off_z * stride_vz + off_h * stride_vh
        O += off_z * stride_oz + off_h * stride_oh  # noqa: E741
        L += (off_z * H + off_h) * M

        if HAS_C2P:
            K_POS += off_z * stride_pk0 + off_h * stride_pk1
        if HAS_P2C:
            Q_POS += off_z * stride_pq0 + off_h * stride_pq1

        offs_m_base = tl.arange(0, BLOCK_M)
        offs_m = start_m * BLOCK_M + offs_m_base
        offs_n_base = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)

        q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
        l_ptrs = L + offs_m

        seq_length = tl.load(SEQ_LENGTHS + off_z).to(tl.int32)
        mask_m = offs_m < seq_length

        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

        m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        offs_n_init = offs_n_base
        k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn)
        v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)

        n_limit = ((seq_length + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        if IS_CAUSAL:
            hi = tl.minimum(n_limit, P_SEQ + (start_m + 1) * BLOCK_M)
            hi = tl.minimum(hi, N)
        else:
            hi = n_limit

        # Log-bucket position constants
        mid_val = NUM_BUCKETS // 2
        inv_log_denom = 1.0 / tl.log((MAX_DISTANCE - 1) / mid_val)

        for start_n in range(0, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + offs_n_base

            mask_n = offs_n < seq_length

            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

            # Content-content attention
            s = tl.zeros([BLOCK_M, BLOCK_N], dtype=input_dtype)
            s += tl.dot(q, k) * sm_scale

            # ── Relative position index computation ──
            relative_positions = offs_m[:, None] - offs_n[None, :]

            if USE_LOG_BUCKET:
                # Log-bucket encoding (DeBERTa v2/v3)
                sign = tl.where(relative_positions > 0.0, 1.0,
                                tl.where(relative_positions < 0.0, -1.0, 0.0))
                abs_relative = tl.abs(relative_positions)
                condition = (relative_positions < mid_val) & (relative_positions > -mid_val)
                abs_pos = tl.where(condition, mid_val - 1.0, abs_relative)
                log_scaled = (tl.log(abs_pos / mid_val)) * inv_log_denom * (mid_val - 1.0)
                log_pos = tl.ceil(log_scaled) + mid_val
                bucket_pos = tl.where(abs_pos <= mid_val, relative_positions, log_pos * sign)
            else:
                # Linear positions (DeBERTa v1)
                bucket_pos = relative_positions

            # ── c2p bias: gather from pos_key ──
            if HAS_C2P:
                c2p_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0),
                                       2 * ATT_SPAN - 1).to(tl.int32)
                k_pos_ptrs = K_POS + offs_m[:, None] * stride_pk2 + c2p_index * stride_pk3
                c2p_bias = tl.load(k_pos_ptrs,
                                   mask=mask_m[:, None] & (c2p_index < 2 * ATT_SPAN),
                                   other=0.0, cache_modifier=".cg")
                s += c2p_bias * sm_scale

            # ── p2c bias: gather from pos_query ──
            if HAS_P2C:
                p2c_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0),
                                       2 * ATT_SPAN - 1).to(tl.int32)
                p2c_index_t = p2c_index.trans(1, 0)
                q_pos_ptrs = Q_POS + offs_n[:, None] * stride_pq2 + p2c_index_t * stride_pq3
                p2c_bias_t = tl.load(q_pos_ptrs,
                                     mask=mask_n[:, None] & (p2c_index_t < 2 * ATT_SPAN),
                                     other=0.0, cache_modifier=".cg")
                p2c_bias = p2c_bias_t.trans(1, 0)
                s += p2c_bias * sm_scale

            # Mask invalid positions
            s = tl.where(mask_n[None, :], s, float("-inf"))

            if IS_CAUSAL:
                causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
                s = tl.where(causal_mask, s, float("-inf"))

            # ── Online softmax ──
            m_i_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2((m_i - m_i_new) * log2e)
            p = tl.math.exp2((s - m_i_new[:, None]) * log2e)
            acc *= alpha[:, None]
            acc += tl.dot(p.to(q.dtype), v)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new

            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn

        # Finalize
        if IS_CAUSAL and LARGER_M:
            is_empty_line = (offs_m + P_SEQ) < 0
            acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
            l = tl.where(is_empty_line, float("-inf"), m_i + tl.log(l_i))  # noqa: E741
        else:
            acc = acc * (1.0 / l_i[:, None])
            l = m_i + tl.log(l_i)  # noqa: E741

        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(q.dtype), mask=mask_m[:, None], cache_modifier=".cg")


# ==============================================================================
# Python Wrappers
# ==============================================================================

def _triton_flash_deberta_fwd(q, k, v, seq_lengths, pos_key, pos_query,
                               causal, sm_scale, position_buckets,
                               max_relative_distance, use_log_bucket):
    """Triton-based forward pass."""
    B, H, M, D = q.shape
    N = k.shape[2]
    P_SEQ = N - M

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    if seq_lengths is None:
        seq_lengths = torch.full((B,), M, dtype=torch.int32, device=q.device)

    has_c2p = pos_key is not None
    has_p2c = pos_query is not None

    ATT_SPAN = position_buckets if position_buckets > 0 else max_relative_distance

    BLOCK_M, BLOCK_N, num_stages, num_warps = _get_fwd_config(
        M, N, D, has_c2p or has_p2c, ATT_SPAN
    )

    larger_m = M > N
    divisible_m = (M % BLOCK_M) == 0
    divisible_n = (N % BLOCK_N) == 0

    grid = (_cdiv(M, BLOCK_M), H, B)
    o = torch.zeros_like(q)
    L = torch.zeros((B, H, M), device=q.device, dtype=torch.float32)

    # Stride helpers for optional tensors
    if has_c2p:
        stride_pk = pos_key.stride()
    else:
        stride_pk = (0, 0, 0, 0)
    if has_p2c:
        stride_pq = pos_query.stride()
    else:
        stride_pq = (0, 0, 0, 0)

    with torch.cuda.device(q.device.index):
        _flash_deberta_fwd_kernel[grid](
            q, k, v,
            pos_key, pos_query,
            L, o,
            seq_lengths,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            stride_pk[0], stride_pk[1], stride_pk[2], stride_pk[3],
            stride_pq[0], stride_pq[1], stride_pq[2], stride_pq[3],
            B, H, M, N, P_SEQ,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal, LARGER_M=larger_m,
            DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
            HAS_C2P=has_c2p, HAS_P2C=has_p2c,
            ATT_SPAN=ATT_SPAN,
            NUM_BUCKETS=position_buckets if position_buckets > 0 else max_relative_distance * 2,
            MAX_DISTANCE=max_relative_distance if max_relative_distance > 0 else 512,
            USE_LOG_BUCKET=use_log_bucket,
            num_warps=num_warps, num_stages=num_stages,
        )

    return o


def _pytorch_flash_deberta_fwd(q, k, v, seq_lengths, pos_key, pos_query,
                                causal, sm_scale, position_buckets,
                                max_relative_distance, use_log_bucket):
    """
    PyTorch fallback: standard matmul attention with disentangled bias.

    Uses the same interface as the Triton kernel for drop-in replacement.
    Input: Q/K/V (B, H, M, D), pos_key (B, H, M, 2*ATT_SPAN), pos_query (B, H, N, 2*ATT_SPAN)
    """
    B, H, M, D = q.shape
    N = k.shape[2]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    ATT_SPAN = position_buckets if position_buckets > 0 else max_relative_distance

    # Content-content
    attn = torch.matmul(q, k.transpose(-1, -2)) * sm_scale

    # Build relative position indices
    q_ids = torch.arange(M, device=q.device)
    k_ids = torch.arange(N, device=q.device)
    relative_pos = q_ids[:, None] - k_ids[None, :]  # (M, N)

    if use_log_bucket and position_buckets > 0 and max_relative_distance > 0:
        # Log-bucket
        sign = torch.sign(relative_pos)
        mid = position_buckets // 2
        abs_pos = torch.where(
            (relative_pos < mid) & (relative_pos > -mid),
            torch.tensor(mid - 1, device=q.device).float(),
            torch.abs(relative_pos).float(),
        )
        log_pos = (
            torch.ceil(
                torch.log(abs_pos / mid) /
                torch.log(torch.tensor((max_relative_distance - 1) / mid, device=q.device).float()) *
                (mid - 1)
            ) + mid
        )
        bucket_pos = torch.where(abs_pos <= mid, relative_pos.float(), log_pos * sign.float())
    else:
        bucket_pos = relative_pos.float()

    # c2p bias
    if pos_key is not None:
        c2p_idx = torch.clamp(bucket_pos + ATT_SPAN, 0, ATT_SPAN * 2 - 1).long()
        c2p_idx = c2p_idx.unsqueeze(0).unsqueeze(0).expand(B, H, M, N)
        c2p_bias = torch.gather(pos_key, dim=-1, index=c2p_idx)
        attn = attn + c2p_bias * sm_scale

    # p2c bias
    if pos_query is not None:
        p2c_idx = torch.clamp(-bucket_pos + ATT_SPAN, 0, ATT_SPAN * 2 - 1).long()
        p2c_idx_for_k = p2c_idx.unsqueeze(0).unsqueeze(0).expand(B, H, N, N)
        # pos_query is (B, H, N, 2*ATT_SPAN), gather along last dim
        p2c_bias = torch.gather(pos_query, dim=-1, index=p2c_idx_for_k)
        attn = attn + p2c_bias.transpose(-1, -2) * sm_scale

    # Mask
    if seq_lengths is not None:
        # Build mask from seq_lengths (B,)
        mask = torch.arange(N, device=q.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        attn = attn.masked_fill(~mask, float("-inf"))

    if causal:
        causal_mask = torch.tril(torch.ones(M, N, device=q.device, dtype=torch.bool))
        attn = attn.masked_fill(~causal_mask, float("-inf"))

    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    return out


# ==============================================================================
# Public API
# ==============================================================================

def flash_deberta_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lengths: Optional[torch.Tensor] = None,
    pos_key: Optional[torch.Tensor] = None,
    pos_query: Optional[torch.Tensor] = None,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    position_buckets: int = 0,
    max_relative_distance: int = 0,
    use_log_bucket: bool = False,
) -> torch.Tensor:
    """
    Flash attention with DeBERTa disentangled relative position bias.

    Args:
        q: Query tensor (B, H, M, D)
        k: Key tensor (B, H, N, D)
        v: Value tensor (B, H, N, D)
        seq_lengths: Per-batch sequence lengths (B,), int32. None = all full length.
        pos_key: Pre-computed Q·pos_key_proj(rel_emb)^T, shape (B, H, M, 2*ATT_SPAN)
        pos_query: Pre-computed K·pos_query_proj(rel_emb)^T, shape (B, H, N, 2*ATT_SPAN)
        causal: Apply causal masking
        sm_scale: Softmax scale (default: 1/sqrt(D))
        position_buckets: Number of position buckets for log-bucket encoding
        max_relative_distance: Maximum relative distance
        use_log_bucket: If True, use log-bucket position encoding (v2/v3).
                        If False, use linear positions (v1).

    Returns:
        Output tensor (B, H, M, D)
    """
    if HAS_TRITON and q.is_cuda:
        return _triton_flash_deberta_fwd(
            q, k, v, seq_lengths, pos_key, pos_query,
            causal, sm_scale, position_buckets, max_relative_distance,
            use_log_bucket,
        )
    else:
        return _pytorch_flash_deberta_fwd(
            q, k, v, seq_lengths, pos_key, pos_query,
            causal, sm_scale, position_buckets, max_relative_distance,
            use_log_bucket,
        )


def clear_config_cache():
    """Clear the kernel configuration cache."""
    _get_fwd_config.cache_clear()


__all__ = ["flash_deberta_attention", "clear_config_cache", "HAS_TRITON"]
