# SPDX-License-Identifier: Apache-2.0
# Inference-only mT5 (encoder-decoder) adapter for vLLM.
#
# Notes
# -----
# - Implements a vLLM-compatible, inference-only stack for mT5 / T5-style models.
# - Uses vLLM's tensor-parallel linear layers and Attention kernel for self-attention.
# - Cross-attention uses the same kernel; encoder K/V are recomputed by default per call but
#   can be cached or preprojected externally if desired.
# - Relative position bias (RPB) is supported by precomputing the bias and passing it into
#   the Attention kernel if the kernel exposes an `attn_bias` argument. If not available
#   in your vLLM version, a safe fallback path computes attention weights in PyTorch for
#   the small query lengths used during decoding.
# - Pipeline parallelism is supported via make_layers / IntermediateTensors.
#
# This file exposes `MT5ForConditionalGeneration` which implements vLLM's SupportsPP contract
# and a weight loader that consumes Hugging Face mt5 checkpoints.

from itertools import islice
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import MT5Config as HFMT5Config

try:
    from vllm.attention.layer import Attention
except ImportError:
    from vllm.attention import Attention
from vllm.config import CacheConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,  # <-- needed for TP head slicing
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.layernorm import RMSNorm as MT5RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)

# SamplingMetadata not needed for encoder-only pooling models in vLLM 0.14.x
try:
    from vllm.model_executor.sampling_metadata import SamplingMetadata
except ImportError:
    SamplingMetadata = None
from vllm.model_executor.models.utils import (
    make_layers,
)
from vllm.sequence import IntermediateTensors

try:
    from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
except ImportError:
    EncoderDecoderModelRunner = None

try:
    from vllm.attention.layer import Attention, AttentionType  # noqa: F401
except ImportError:
    pass

from kernels.ff_fused import fused_gelu_mul_dropout
from kernels.flash_attention_rpb import flash_attention_rpb_final


def _make_contiguous_additive_bias(
    rpb: Optional[torch.Tensor],
    padding_mask: Optional[torch.Tensor],
    batch_size: int,
    num_heads: int,
    q_len: int,
    k_len: int,
    dtype: torch.dtype,
    device: torch.device,
    # Optional: A reusable buffer to avoid re-allocation
    bias_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Creates a dense, contiguous, additive attention bias tensor to guarantee
    that the memory-efficient SDPA kernel can be used.
    """
    if (bias_buffer is None
            or bias_buffer.shape != (batch_size, num_heads, q_len, k_len)
            or bias_buffer.dtype != dtype
            or bias_buffer.device != device):
        bias = torch.empty((batch_size, num_heads, q_len, k_len),
                           dtype=dtype,
                           device=device)
    else:
        bias = bias_buffer

    if rpb is not None:
        # .copy_() with .expand() broadcasts and forces a contiguous materialization
        bias.copy_(rpb.expand(batch_size, -1, -1, -1))
    else:
        bias.zero_()

    if padding_mask is not None:
        bias.add_(padding_mask)

    return bias


# -----------------------------
# Relative Position Bias (RPB)
# -----------------------------
class T5RelativePositionBias(nn.Module):
    def __init__(self, num_buckets: int, max_distance: int, n_heads: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor,
                                  bidirectional: bool,
                                  num_buckets: int,
                                  max_distance: int) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / torch.log(torch.tensor(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets = torch.where(is_small, relative_position, relative_position_if_large) + relative_buckets
        return relative_buckets

    def forward(self, q_len: int, k_len: int, device, *, bidirectional: bool,
                cache_position: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cache_position is None:
            context_position = torch.arange(q_len, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        memory_position = torch.arange(k_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # (q_len, k_len)
        rp_bucket = self._relative_position_bucket(
            relative_position, bidirectional=bidirectional,
            num_buckets=self.num_buckets, max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)  # (q_len, k_len, n_heads)
        values = values.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, q_len, k_len)
        return values


# -----------------------------
# Attention blocks
# -----------------------------


class T5SelfAttention(nn.Module):
    # __init__ and _get_rpb_bias methods are unchanged...
    def __init__(self, d_model, n_heads, d_kv, dropout, cache_config, quant_config,
                 prefix, use_rpb, rpb_num_buckets, rpb_max_distance, is_decoder):
        super().__init__()
        self.is_decoder = is_decoder
        self.head_dim = d_kv
        self.dropout = dropout

        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_n_heads = n_heads
        self.n_heads = n_heads // self.tp_size
        self.inner_dim = self.n_heads * self.head_dim

        self.q_proj = ColumnParallelLinear(d_model, self.inner_dim, bias=False,
                                           quant_config=quant_config, prefix=f"{prefix}.q")
        self.k_proj = ColumnParallelLinear(d_model, self.inner_dim, bias=False,
                                           quant_config=quant_config, prefix=f"{prefix}.k")
        self.v_proj = ColumnParallelLinear(d_model, self.inner_dim, bias=False,
                                           quant_config=quant_config, prefix=f"{prefix}.v")
        self.out_proj = RowParallelLinear(self.inner_dim, d_model, bias=False,
                                          quant_config=quant_config, prefix=f"{prefix}.o")

        self.use_rpb = use_rpb
        if use_rpb:
            self.rpb = T5RelativePositionBias(rpb_num_buckets, rpb_max_distance, self.total_n_heads)
        else:
            self.rpb = None

        self._rpb_cache = {}
        self._bias_buf = None

    def _local_rpb_table(self, shared):
        if self.tp_size > 1:
            tp_rank = get_tensor_model_parallel_rank()
            start = tp_rank * self.n_heads
            end = (tp_rank + 1) * self.n_heads
            return shared.relative_attention_bias.weight[:, start:end].contiguous()
        return shared.relative_attention_bias.weight

    def _get_rpb_bias(self, q_len, k_len, device, dtype):
        if not self.use_rpb or self.rpb is None:
            return None
        tp_rank = get_tensor_model_parallel_rank()
        key = (q_len, k_len, device, dtype, tp_rank)
        if key in self._rpb_cache:
            return self._rpb_cache[key]

        full_bias = self.rpb(q_len, k_len, device,
                             bidirectional=not self.is_decoder)
        if full_bias.dtype != dtype:
            full_bias = full_bias.to(dtype)

        if self.tp_size > 1:
            start = tp_rank * self.n_heads
            end = (tp_rank + 1) * self.n_heads
            bias = full_bias[:, start:end, :, :]
        else:
            bias = full_bias
        self._rpb_cache[key] = bias
        return bias

    def forward(self,
                hidden_states: torch.Tensor,
                mask: Optional[torch.Tensor] = None,           # Legacy arg
                position_bias: Optional[torch.Tensor] = None,  # Legacy arg
                attention_mask: Optional[torch.Tensor] = None, # Optimized mask
                rpb_table: Optional[torch.Tensor] = None,      # Optimized table
                attn_metadata=None,
                **_):

        B, L, _ = hidden_states.shape

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

        use_triton = True
        try:
            _ = flash_attention_rpb_final
        except NameError:
            use_triton = False

        if use_triton:
            num_buckets = 32
            max_distance = 128
            if self.rpb is not None:
                num_buckets = self.rpb.num_buckets
                max_distance = self.rpb.max_distance
            elif rpb_table is not None:
                # Standard HF Layout: [Buckets, Heads] -> shape[0] is buckets
                num_buckets = rpb_table.shape[0]

            attn_ctx = flash_attention_rpb_final(
                q, k, v,
                rpb_table=rpb_table,
                num_buckets=num_buckets,
                max_distance=max_distance,
                is_decoder=bool(self.is_decoder),
                padding_mask=attention_mask,
                dropout_p=0.0,
                softmax_scale=1.0,
                use_autotune=True,
            )

            attn_out = attn_ctx.transpose(1, 2).contiguous().view(B, L, self.inner_dim)
            out, _ = self.out_proj(attn_out)
            return out

        # Fallback (SDPA) - Safe logic to reconstruct dense bias if needed
        rpb_dense = position_bias
        if rpb_dense is None and rpb_table is not None:
            rpb_dense = self._get_rpb_bias(q_len=L, k_len=L, device=q.device, dtype=q.dtype)

        add_mask = mask
        if add_mask is None and attention_mask is not None:
            add_mask = attention_mask

        attn_bias = None
        if rpb_dense is not None or add_mask is not None:
            attn_bias = _make_contiguous_additive_bias(
                rpb=rpb_dense,
                padding_mask=add_mask,
                batch_size=B,
                num_heads=self.n_heads,
                q_len=L,
                k_len=L,
                dtype=q.dtype,
                device=q.device,
                bias_buffer=self._bias_buf,
            )
            self._bias_buf = attn_bias

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
            ctx = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=1.0,
            )

        attn_out = ctx.transpose(1, 2).contiguous().view(B, L, self.inner_dim)
        out, _ = self.out_proj(attn_out)
        return out



def quantization_config_or_none(qc: Optional[QuantizationConfig]) -> Optional[QuantizationConfig]:
    # Helper for clarity; vLLM linear layers accept None when no quantization is active.
    return qc if isinstance(qc, QuantizationConfig) else None


# -----------------------------
# Feed-forward (Dense or Gated)
# -----------------------------
class T5FF(nn.Module):
    """Optimized T5/mT5 feed-forward block.

    Strategy:
      - Keep matmuls on vLLM Column/RowParallelLinear (cuBLASLt).
      - Fuse GELU ∘ MUL ∘ (optional) Dropout via Triton (elementwise only).
      - Works with or without vLLM; falls back to torch.nn.Linear otherwise.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        act_fn: str,
        dropout: float,
        quant_config: Optional[object],
        prefix: str,
        gated: bool,
    ):
        super().__init__()
        self.gated = bool(gated)
        self.dropout_p = float(dropout)
        self.act_fn_name = act_fn
        self.quant_config = quant_config

        # Try vLLM tensor-parallel linears; else fall back to nn.Linear
        try:
            from vllm.model_executor.layers.activation import get_act_fn as _vllm_get_act
            from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
            self._use_vllm = True
            CPL = ColumnParallelLinear
            RPL = RowParallelLinear
            self._get_act = _vllm_get_act
        except Exception:
            self._use_vllm = False
            CPL = nn.Linear
            RPL = nn.Linear
            def _fallback_get_act(name: str):
                n = name.lower()
                if n in ("gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"):
                    return lambda x: F.gelu(x, approximate="tanh")
                if n == "gelu_accurate":
                    return lambda x: F.gelu(x, approximate="none")
                if n == "relu":
                    return F.relu
                if n in ("silu", "swish"):
                    return F.silu
                return lambda x: F.gelu(x, approximate="tanh")
            self._get_act = _fallback_get_act

        if self.gated:
            # T5 gating: wi_0 -> activation, wi_1 -> linear, then elementwise product
            self.wi_0 = CPL(d_model, d_ff, bias=False,
                            quant_config=quant_config if self._use_vllm else None,
                            prefix=f"{prefix}.wi_0" if self._use_vllm else None)
            self.wi_1 = CPL(d_model, d_ff, bias=False,
                            quant_config=quant_config if self._use_vllm else None,
                            prefix=f"{prefix}.wi_1" if self._use_vllm else None)
            self.wo   = RPL(d_ff, d_model, bias=False,
                            quant_config=quant_config if self._use_vllm else None,
                            prefix=f"{prefix}.wo"   if self._use_vllm else None)
        else:
            self.wi = CPL(d_model, d_ff, bias=False,
                          quant_config=quant_config if self._use_vllm else None,
                          prefix=f"{prefix}.wi" if self._use_vllm else None)
            self.wo = RPL(d_ff, d_model, bias=False,
                          quant_config=quant_config if self._use_vllm else None,
                          prefix=f"{prefix}.wo" if self._use_vllm else None)
            self.act = self._get_act(self.act_fn_name)

        # Enable elementwise fusion only if not quantized and gated
        self.use_elementwise_fusion = (self.quant_config is None) and self.gated

    @staticmethod
    def _apply_linear(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """vLLM linears may return (out, bias) tuples; unwrap if needed."""
        y = layer(x)
        return y[0] if isinstance(y, tuple) else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            # Two GEMMs via cuBLASLt (vLLM linears)
            gate = self._apply_linear(self.wi_0, x)  # [B,L, d_ff_local]
            up   = self._apply_linear(self.wi_1, x)  # [B,L, d_ff_local]

            if self.use_elementwise_fusion:
                hidden = fused_gelu_mul_dropout(
                    gate=gate,
                    up=up,
                    act_fn=self.act_fn_name,
                    dropout_p=self.dropout_p,
                    training=self.training,
                )
            else:
                # Fallback: separate ops (kept for quantized paths)
                if self.act_fn_name.lower() in ("gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"):
                    gate = F.gelu(gate, approximate="tanh")
                elif self.act_fn_name.lower() == "gelu_accurate":
                    gate = F.gelu(gate, approximate="none")
                elif self.act_fn_name.lower() == "relu":
                    gate = F.relu(gate)
                elif self.act_fn_name.lower() in ("silu", "swish"):
                    gate = F.silu(gate)
                else:
                    gate = F.gelu(gate, approximate="tanh")
                hidden = gate * up
                if self.dropout_p > 0.0 and self.training:
                    hidden = F.dropout(hidden, p=self.dropout_p)
        else:
            hidden = self._apply_linear(self.wi, x)
            hidden = self.act(hidden)
            if self.dropout_p > 0.0 and self.training:
                hidden = F.dropout(hidden, p=self.dropout_p)

        out = self._apply_linear(self.wo, hidden)
        return out



# -----------------------------
# Encoder/Decoder layers
# -----------------------------
class T5EncoderLayer(nn.Module):
    def __init__(self, cfg: HFMT5Config,
                 cache_config: Optional[CacheConfig],
                 quant_config: Optional[QuantizationConfig],
                 prefix: str,
                 has_rpb: bool = True):  # Add this parameter
        super().__init__()
        self.ln1 = MT5RMSNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.self_attn = T5SelfAttention(
            d_model=cfg.d_model, n_heads=cfg.num_heads, d_kv=cfg.d_kv,
            dropout=cfg.dropout_rate, cache_config=cache_config,
            quant_config=quant_config, prefix=f"{prefix}.SelfAttention",
            use_rpb=has_rpb,  # Use the parameter here
            rpb_num_buckets=cfg.relative_attention_num_buckets,
            rpb_max_distance=cfg.relative_attention_max_distance,
            is_decoder=False,
        )
        self.ln2 = MT5RMSNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.ff = T5FF(cfg.d_model, cfg.d_ff, cfg.dense_act_fn,
                       dropout=cfg.dropout_rate, quant_config=quant_config,
                       prefix=f"{prefix}.FF", gated=cfg.is_gated_act)

    def forward(self,
                x: torch.Tensor,
                attn_metadata=None,
                rpb_table: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        h = x + self.self_attn(
            self.ln1(x),
            attn_metadata=attn_metadata,
            rpb_table=rpb_table,
            attention_mask=attention_mask
        )

        h = h + self.ff(self.ln2(h))
        return h

# -----------------------------
# Encoder/Decoder stacks
# -----------------------------
class MT5Encoder(nn.Module):
    """Encoder-only stack for GLiNER span classification."""
    def __init__(self, cfg: HFMT5Config,
                 cache_config,
                 quant_config,
                 prefix: str):
        super().__init__()
        self.config = cfg
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.d_model)
        self.dropout = cfg.dropout_rate

        def create_layer(prefix: str) -> T5EncoderLayer:
            layer_idx = int(prefix.split('.')[-1])
            return T5EncoderLayer(
                cfg, cache_config, quant_config, prefix,
                has_rpb=(layer_idx == 0)
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            cfg.num_layers,
            create_layer,
            prefix=f"{prefix}.block",
        )
        self.final_ln = MT5RMSNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                inputs_embeds: Optional[torch.Tensor] = None,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                attn_metadata=None) -> Union[torch.Tensor, IntermediateTensors]:

        if get_pp_group().is_first_rank:
            x = self.get_input_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
            if self.dropout and self.training:
                x = torch.nn.functional.dropout(x, p=self.dropout)
        else:
            assert intermediate_tensors is not None
            x = intermediate_tensors["hidden_states"]

        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # ------------------------------------------------------------------
        # FIX: Extract RPB Table [Buckets, Heads] - NO TRANSPOSE
        # ------------------------------------------------------------------
        rpb_table_weight = None

        first_layer_attn = self.layers[self.start_layer].self_attn
        if first_layer_attn.rpb is not None:
            # 1. Get raw weights [Buckets, Heads]
            full_weight = first_layer_attn.rpb.relative_attention_bias.weight

            # 2. Handle Tensor Parallelism (TP) slicing
            #    Original HF weights are [Buckets, Heads], so we slice dim 1 (Heads).
            if first_layer_attn.tp_size > 1:
                tp_rank = get_tensor_model_parallel_rank()
                n_heads_local = first_layer_attn.n_heads
                start = tp_rank * n_heads_local
                end = start + n_heads_local
                rpb_table_weight = full_weight[:, start:end]
            else:
                rpb_table_weight = full_weight

            # 3. Ensure contiguous memory.
            #    We do NOT transpose. The kernel will handle the stride.
            rpb_table_weight = rpb_table_weight.to(x.device).contiguous()

        # ------------------------------------------------------------------
        # Prepare Mask [Batch, 1, 1, Seq]
        # ------------------------------------------------------------------
        broadcast_mask = None
        if attention_mask is not None:
            # [Batch, Seq] -> [Batch, 1, 1, Seq]
            broadcast_mask = attention_mask[:, None, None, :]
            broadcast_mask = broadcast_mask.to(dtype=x.dtype)
            broadcast_mask = (1.0 - broadcast_mask) * torch.finfo(x.dtype).min

        # ------------------------------------------------------------------
        # Pass Down
        # ------------------------------------------------------------------
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            x = layer(
                x,
                attn_metadata=attn_metadata,
                rpb_table=rpb_table_weight,
                attention_mask=broadcast_mask
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": x})

        x = self.final_ln(x)
        if self.dropout and self.training:
            x = torch.nn.functional.dropout(x, p=self.dropout)

        if len(original_shape) == 2:
            x = x.squeeze(0)

        return x



# Debug logging disabled - uncomment to re-enable
# import vllm.worker.model_runner as model_runner
# original_prepare_model_inputs = model_runner.ModelRunner.prepare_model_input

# def debug_prepare_model_input(self, *args, **kwargs):
#     result = original_prepare_model_inputs(self, *args, **kwargs)
#     print(f"[INTERCEPTED] Model inputs: {result.keys() if hasattr(result, 'keys') else type(result)}")
#     return result

# model_runner.ModelRunner.prepare_model_input = debug_prepare_model_input
