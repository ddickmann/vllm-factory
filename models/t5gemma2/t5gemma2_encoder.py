"""T5Gemma2 encoder backbone for vLLM Factory."""

from __future__ import annotations

import os
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
except ImportError:
    ROPE_INIT_FUNCTIONS = {}

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import GeluAndMul, get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.model_executor.models.utils import make_layers
from vllm.sequence import IntermediateTensors

from kernels.fused_gemma_rms_norm_dropout_residual import (
    fused_gemma_rms_norm_dropout_residual,
)
from kernels.fused_rope_global import fused_rope_global_apply

from .config import (
    T5Gemma2Config,
    T5Gemma2DecoderConfig,
    T5Gemma2EncoderConfig,
    T5Gemma2TextConfig,
    get_t5gemma2_text_config,
)


def _t5gemma2_use_optimized_paths() -> bool:
    flag = os.getenv("VLLM_FACTORY_T5GEMMA2_REFERENCE_PATH", "")
    return flag.lower() not in {"1", "true", "yes", "on"}


def _resolve_model_config(
    config_or_vllm_config: VllmConfig
    | T5Gemma2Config
    | T5Gemma2EncoderConfig
    | T5Gemma2DecoderConfig
    | T5Gemma2TextConfig,
    *,
    is_encoder: bool,
) -> tuple[
    T5Gemma2Config | T5Gemma2EncoderConfig | T5Gemma2DecoderConfig | T5Gemma2TextConfig,
    T5Gemma2TextConfig | T5Gemma2DecoderConfig,
    CacheConfig | None,
    QuantizationConfig | None,
]:
    """Normalize raw HF configs and `VllmConfig` into one tuple."""

    if isinstance(config_or_vllm_config, VllmConfig):
        outer_config = config_or_vllm_config.model_config.hf_config
        if isinstance(outer_config, T5Gemma2Config):
            text_config = get_t5gemma2_text_config(outer_config, is_encoder=is_encoder)
        elif is_encoder and isinstance(outer_config, T5Gemma2EncoderConfig):
            text_config = outer_config.text_config
        elif (not is_encoder) and isinstance(outer_config, T5Gemma2DecoderConfig):
            text_config = outer_config
        else:
            text_config = outer_config
        return (
            outer_config,
            text_config,
            getattr(config_or_vllm_config, "cache_config", None),
            getattr(config_or_vllm_config, "quant_config", None),
        )

    if isinstance(config_or_vllm_config, T5Gemma2Config):
        return (
            config_or_vllm_config,
            get_t5gemma2_text_config(config_or_vllm_config, is_encoder=is_encoder),
            None,
            None,
        )

    if is_encoder and isinstance(config_or_vllm_config, T5Gemma2EncoderConfig):
        return config_or_vllm_config, config_or_vllm_config.text_config, None, None

    return config_or_vllm_config, config_or_vllm_config, None, None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key states."""

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat grouped KV heads into per-query heads."""

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_kv_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _bool_to_additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert an allowed-attention boolean mask into an additive mask."""

    min_value = torch.finfo(dtype).min
    return torch.where(mask, torch.zeros((), device=mask.device, dtype=dtype), min_value)


def _build_segment_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Detect packed-sequence boundaries from position resets."""

    segment_ids = torch.zeros_like(position_ids)
    segment_ids[:, 0] = 1
    if position_ids.shape[1] > 1:
        boundaries = position_ids[:, 1:] <= position_ids[:, :-1]
        segment_ids[:, 1:] = boundaries.to(segment_ids.dtype)
    return segment_ids.cumsum(dim=1)


def _build_attention_segments(
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> list[list[tuple[int, int]]]:
    valid_lengths = attention_mask.to(dtype=torch.int64).sum(dim=1)
    segment_ids = _build_segment_ids(position_ids)
    all_segments: list[list[tuple[int, int]]] = []

    for batch_idx in range(attention_mask.shape[0]):
        valid_len = int(valid_lengths[batch_idx].item())
        if valid_len == 0:
            all_segments.append([])
            continue

        row_segment_ids = segment_ids[batch_idx, :valid_len]
        boundaries = torch.nonzero(
            row_segment_ids[1:] != row_segment_ids[:-1],
            as_tuple=False,
        ).flatten()
        starts = [0, *[int(boundary.item()) + 1 for boundary in boundaries]]
        ends = [*[int(boundary.item()) + 1 for boundary in boundaries], valid_len]
        all_segments.append(list(zip(starts, ends)))

    return all_segments


def _build_sliding_window_mask(
    seq_len: int,
    sliding_window: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    rows = torch.arange(seq_len, device=device)
    dist = rows[:, None] - rows[None, :]
    left_window = (sliding_window + 1) // 2
    right_window = (sliding_window // 2) + 1
    window_mask = ((dist >= 0) & (dist < left_window)) | (
        (dist < 0) & (-dist < right_window)
    )
    return window_mask[None, None, :, :]


def _build_encoder_attention_plans(
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    sliding_window: int | None,
) -> dict[str, dict[str, torch.Tensor | list[list[tuple[int, int]]] | None]]:
    """Build optimized attention metadata without dense packed masks."""

    attention_mask = attention_mask.to(dtype=torch.bool)
    all_tokens_valid = bool(attention_mask.all())
    has_position_resets = bool(
        position_ids.shape[1] > 1
        and (position_ids[:, 1:] <= position_ids[:, :-1]).any()
    )
    segments = (
        None
        if (all_tokens_valid and not has_position_resets)
        else _build_attention_segments(attention_mask, position_ids)
    )

    full_attention_mask = None
    if segments is None and not all_tokens_valid:
        full_attention_mask = attention_mask[:, None, None, :]

    sliding_attention_mask = full_attention_mask
    if sliding_window is not None:
        sliding_attention_mask = _build_sliding_window_mask(
            attention_mask.shape[1],
            sliding_window,
            device=attention_mask.device,
        )

    return {
        "full_attention": {
            "mask": full_attention_mask,
            "segments": segments,
        },
        "sliding_attention": {
            "mask": sliding_attention_mask,
            "segments": segments,
        },
    }


def _build_encoder_attention_masks_reference(
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    sliding_window: int | None,
) -> dict[str, torch.Tensor]:
    """Build HF-style encoder masks for the reference path."""

    attention_mask = attention_mask.to(dtype=torch.bool)
    pair_mask = attention_mask[:, :, None] & attention_mask[:, None, :]

    segment_ids = _build_segment_ids(position_ids)
    block_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
    full_mask = pair_mask & block_mask

    mask_mapping = {"full_attention": full_mask[:, None, :, :]}

    if sliding_window is not None:
        window_mask = _build_sliding_window_mask(
            attention_mask.shape[1],
            sliding_window,
            device=attention_mask.device,
        )
        mask_mapping["sliding_attention"] = (
            full_mask[:, None, :, :] & window_mask
        )
    else:
        mask_mapping["sliding_attention"] = mask_mapping["full_attention"]
    return mask_mapping


def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    softcap: float | None,
    training: bool,
) -> torch.Tensor:
    """Fallback attention path when softcapping is active."""

    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if softcap is not None:
        attn_weights = torch.tanh(attn_weights / softcap) * softcap

    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attention_mask, torch.finfo(query.dtype).min)
        else:
            attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=training)
    return torch.matmul(attn_weights, value)


class T5Gemma2TextScaledWordEmbedding(VocabParallelEmbedding):
    """Embedding layer with T5Gemma2 scaling and EOI replacement."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        embed_scale: float,
        eoi_token_index: int | None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.scalar_embed_scale = float(embed_scale)
        self.register_buffer(
            "embed_scale",
            torch.tensor(self.scalar_embed_scale),
            persistent=False,
        )
        self.eoi_token_index = eoi_token_index
        self.eoi_embedding = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = super().forward(input_ids)
        embeddings = embeddings * self.embed_scale.to(dtype=embeddings.dtype, device=embeddings.device)
        if self.eoi_token_index is not None:
            eoi_mask = input_ids == self.eoi_token_index
            embeddings = torch.where(
                eoi_mask.unsqueeze(-1),
                self.eoi_embedding.to(dtype=embeddings.dtype, device=embeddings.device),
                embeddings,
            )
        return embeddings


class T5Gemma2MultiModalProjector(nn.Module):
    """Project SigLIP features into the encoder text space."""

    def __init__(self, config: T5Gemma2Config | T5Gemma2EncoderConfig) -> None:
        super().__init__()
        encoder_config = config.encoder if isinstance(config, T5Gemma2Config) else config
        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(
                encoder_config.vision_config.hidden_size,
                encoder_config.text_config.hidden_size,
            )
        )
        self.mm_soft_emb_norm = GemmaRMSNorm(
            encoder_config.vision_config.hidden_size,
            eps=encoder_config.vision_config.layer_norm_eps,
        )
        self.patches_per_image = int(
            encoder_config.vision_config.image_size
            // encoder_config.vision_config.patch_size
        )
        self.tokens_per_side = int(encoder_config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
        )

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_length = vision_outputs.shape
        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size,
            seq_length,
            self.patches_per_image,
            self.patches_per_image,
        ).contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2).transpose(1, 2)
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)
        projected_vision_outputs = torch.matmul(
            normed_vision_outputs,
            self.mm_input_projection_weight,
        )
        return projected_vision_outputs.type_as(vision_outputs)


class T5Gemma2RotaryEmbedding(nn.Module):
    """Per-layer-type RoPE for T5Gemma2."""

    def __init__(self, config: T5Gemma2TextConfig):
        super().__init__()
        self.config = config
        self.layer_types = sorted(set(config.layer_types))

        for layer_type in self.layer_types:
            rope_params = config.rope_parameters.get(layer_type)
            if rope_params is None:
                continue

            rope_type = rope_params.get("rope_type", "default")
            if rope_type != "default" and rope_type in ROPE_INIT_FUNCTIONS:
                inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](
                    config,
                    device=None,
                    layer_type=layer_type,
                )
            else:
                inv_freq, attention_scaling = self.compute_default_rope_parameters(
                    config,
                    layer_type=layer_type,
                )

            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: T5Gemma2TextConfig,
        *,
        layer_type: str,
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        return inv_freq, 1.0

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq").to(device=x.device)
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class T5Gemma2MLP(nn.Module):
    """Gemma-style gated MLP backed by vLLM parallel linear layers."""

    def __init__(
        self,
        config: T5Gemma2TextConfig | T5Gemma2DecoderConfig,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_activation == "gelu_pytorch_tanh":
            self.act_fn = GeluAndMul(approximate="tanh")
        else:
            self.act_fn = get_act_and_mul_fn(config.hidden_activation)
        self.dropout = nn.Dropout(getattr(config, "dropout_rate", 0.0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states = self.dropout(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class T5Gemma2SelfAttention(nn.Module):
    """HF-exact self-attention using vLLM linears and norms."""

    def __init__(
        self,
        config: T5Gemma2TextConfig | T5Gemma2DecoderConfig,
        *,
        layer_idx: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.head_dim = config.head_dim
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads = self.total_num_heads // self.tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.num_queries_per_kv = self.total_num_heads // self.total_num_kv_heads
        self.scaling = float(config.query_pre_attn_scalar) ** -0.5
        self.attn_softcap = config.attn_logit_softcapping
        self.dropout_p = getattr(config, "attention_dropout", 0.0)

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        enable_gqa: bool,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            scale=self.scaling,
            enable_gqa=enable_gqa,
        )

    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        k = repeat_kv(k, self.num_queries_per_kv)
        v = repeat_kv(v, self.num_queries_per_kv)

        if self.attn_softcap is None:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
                scale=self.scaling,
            )
        else:
            attn_output = eager_attention_forward(
                q,
                k,
                v,
                attention_mask=attention_mask,
                dropout=self.dropout_p if self.training else 0.0,
                scaling=self.scaling,
                softcap=self.attn_softcap,
                training=self.training,
            )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        attn_output, _ = self.o_proj(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: (
            torch.Tensor
            | dict[str, torch.Tensor | list[list[tuple[int, int]]] | None]
            | None
        ),
    ) -> torch.Tensor:
        if (not _t5gemma2_use_optimized_paths()) or self.attn_softcap is not None:
            return self._forward_reference(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        batch_size, seq_len, _ = hidden_states.shape
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = fused_rope_global_apply(q, k, *position_embeddings)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        assert isinstance(attention_mask, dict)
        shared_mask = attention_mask.get("mask")
        segments = attention_mask.get("segments")
        enable_gqa = self.num_heads > self.num_kv_heads

        if segments is None:
            attn_output = self._scaled_dot_product_attention(
                q,
                k,
                v,
                attention_mask=shared_mask,
                enable_gqa=enable_gqa,
            )
        else:
            attn_output = torch.zeros(
                (batch_size, self.num_heads, seq_len, self.head_dim),
                device=q.device,
                dtype=q.dtype,
            )
            for batch_idx, batch_segments in enumerate(segments):
                for start, end in batch_segments:
                    segment_mask = None
                    if shared_mask is not None:
                        segment_mask = shared_mask[..., : end - start, : end - start]
                    attn_output[batch_idx : batch_idx + 1, :, start:end, :] = (
                        self._scaled_dot_product_attention(
                            q[batch_idx : batch_idx + 1, :, start:end, :],
                            k[batch_idx : batch_idx + 1, :, start:end, :],
                            v[batch_idx : batch_idx + 1, :, start:end, :],
                            attention_mask=segment_mask,
                            enable_gqa=enable_gqa,
                        )
                    )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        attn_output, _ = self.o_proj(attn_output)
        return attn_output


class T5Gemma2EncoderLayer(nn.Module):
    """One HF-exact T5Gemma2 encoder block."""

    def __init__(
        self,
        config: T5Gemma2TextConfig,
        *,
        layer_idx: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = T5Gemma2SelfAttention(
            config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.pre_self_attn_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_self_attn_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp = T5Gemma2MLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.dropout = nn.Dropout(getattr(config, "dropout_rate", 0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: (
            torch.Tensor
            | dict[str, torch.Tensor | list[list[tuple[int, int]]] | None]
            | None
        ),
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        if _t5gemma2_use_optimized_paths():
            hidden_states = fused_gemma_rms_norm_dropout_residual(
                hidden_states,
                residual,
                self.post_self_attn_layernorm.weight,
                eps=self.post_self_attn_layernorm.variance_epsilon,
                dropout_p=self.dropout.p,
                training=self.training,
            )
        else:
            hidden_states = self.post_self_attn_layernorm(hidden_states)
            hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if _t5gemma2_use_optimized_paths():
            hidden_states = fused_gemma_rms_norm_dropout_residual(
                hidden_states,
                residual,
                self.post_feedforward_layernorm.weight,
                eps=self.post_feedforward_layernorm.variance_epsilon,
                dropout_p=self.dropout.p,
                training=self.training,
            )
        else:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


class T5Gemma2Encoder(nn.Module):
    """Text or multimodal encoder backbone for T5Gemma2."""

    def __init__(
        self,
        config_or_vllm_config: VllmConfig | T5Gemma2Config | T5Gemma2EncoderConfig | T5Gemma2TextConfig,
        *,
        prefix: str = "encoder",
        shared_embed_tokens: T5Gemma2TextScaledWordEmbedding | None = None,
        shared_eoi_embedding: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        outer_config, text_config, cache_config, quant_config = _resolve_model_config(
            config_or_vllm_config,
            is_encoder=True,
        )
        del cache_config  # The encoder path does not use paged KV cache.
        self.outer_config = outer_config
        self.config = text_config
        self.quant_config = quant_config
        eoi_token_index = getattr(outer_config, "eoi_token_index", None)
        self.use_optimized_attention = (
            _t5gemma2_use_optimized_paths()
            and text_config.attn_logit_softcapping is None
        )

        encoder_config: T5Gemma2EncoderConfig | None
        if isinstance(outer_config, T5Gemma2Config):
            encoder_config = outer_config.encoder
        elif isinstance(outer_config, T5Gemma2EncoderConfig):
            encoder_config = outer_config
        else:
            encoder_config = None

        self.encoder_config = encoder_config
        self.image_token_index = getattr(encoder_config, "image_token_index", None)
        text_prefix = f"{prefix}.text_model"

        if shared_embed_tokens is not None:
            self.embed_tokens = shared_embed_tokens
        else:
            self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
                text_config.vocab_size,
                text_config.hidden_size,
                embed_scale=text_config.hidden_size**0.5,
                eoi_token_index=eoi_token_index,
                quant_config=quant_config,
                prefix=f"{text_prefix}.embed_tokens",
            )
        if shared_eoi_embedding is not None:
            self.embed_tokens.eoi_embedding = shared_eoi_embedding

        self.vision_tower: SiglipVisionModel | None = None
        self.multi_modal_projector: T5Gemma2MultiModalProjector | None = None
        if encoder_config is not None:
            self.vision_tower = SiglipVisionModel(
                encoder_config.vision_config,
                quant_config=quant_config,
                prefix=f"{prefix}.vision_tower",
            )
            self.multi_modal_projector = T5Gemma2MultiModalProjector(encoder_config)

        self.dropout = nn.Dropout(getattr(text_config, "dropout_rate", 0.0))
        self.rotary_emb = T5Gemma2RotaryEmbedding(text_config)
        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config.num_hidden_layers,
            lambda prefix: T5Gemma2EncoderLayer(
                text_config,
                layer_idx=int(prefix.split(".")[-1]),
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{text_prefix}.layers",
        )
        self.norm = GemmaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def get_input_embeddings(self) -> T5Gemma2TextScaledWordEmbedding:
        return self.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.vision_tower is None or self.multi_modal_projector is None:
            raise ValueError("This encoder configuration does not have a vision tower.")
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        vision_outputs = self.vision_tower(pixel_values)
        return self.multi_modal_projector(vision_outputs)

    def _get_image_placeholder_mask(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if self.image_token_index is None:
            raise ValueError("image_token_index must be configured for multimodal inputs.")

        if input_ids is not None:
            return input_ids == self.image_token_index

        placeholder_ids = torch.full(
            (1,),
            self.image_token_index,
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        placeholder_embedding = self.embed_tokens(placeholder_ids)[0].to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        return inputs_embeds.eq(placeholder_embedding).all(dim=-1)

    def _merge_image_features(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.get_image_features(pixel_values).to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        image_mask = self._get_image_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        expected_elements = int(image_mask.sum().item()) * inputs_embeds.shape[-1]
        if expected_elements != image_features.numel():
            raise ValueError(
                "The number of image placeholder tokens does not match the "
                "projected image features."
            )

        merged_inputs = inputs_embeds.clone()
        merged_inputs.masked_scatter_(
            image_mask.unsqueeze(-1).expand_as(merged_inputs),
            image_features.reshape(-1),
        )
        return merged_inputs

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
        added_batch = False
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            added_batch = True
        if inputs_embeds is not None and inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
            added_batch = True
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_len = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        return input_ids, attention_mask, position_ids, inputs_embeds, added_batch

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        **_: object,
    ) -> torch.Tensor | IntermediateTensors:
        input_ids, attention_mask, position_ids, inputs_embeds, added_batch = self._prepare_inputs(
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
        )

        if get_pp_group().is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids)
                if inputs_embeds is None
                else inputs_embeds
            )
            if pixel_values is not None:
                hidden_states = self._merge_image_features(
                    input_ids=input_ids,
                    inputs_embeds=hidden_states,
                    pixel_values=pixel_values,
                )
            hidden_states = self.dropout(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_optimized_attention:
            mask_mapping = _build_encoder_attention_plans(
                attention_mask,
                position_ids,
                self.config.sliding_window,
            )
        else:
            mask_mapping = _build_encoder_attention_masks_reference(
                attention_mask,
                position_ids,
                self.config.sliding_window,
            )
        position_embeddings = {
            layer_type: self.rotary_emb(
                hidden_states,
                position_ids,
                layer_type=layer_type,
            )
            for layer_type in set(self.config.layer_types)
        }

        for layer_idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer), start=self.start_layer):
            layer_type = self.config.layer_types[layer_idx]
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=mask_mapping[layer_type],
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if added_batch:
            hidden_states = hidden_states.squeeze(0)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load HF-formatted encoder weights with merged MLP projections."""

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        text_and_mm_weights: list[tuple[str, torch.Tensor]] = []
        vision_weights: list[tuple[str, torch.Tensor]] = []
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if name.startswith("vision_tower."):
                vision_weights.append((name[len("vision_tower.") :], loaded_weight))
                continue
            if name.startswith("text_model."):
                name = name[len("text_model.") :]
            text_and_mm_weights.append((name, loaded_weight))

        for name, loaded_weight in text_and_mm_weights:
            if "rotary_emb." in name:
                continue

            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                mapped_name = name.replace(shard_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    continue
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if self.vision_tower is not None and vision_weights:
            for name in self.vision_tower.load_weights(vision_weights):
                loaded_params.add(f"vision_tower.{name}")

        return loaded_params


__all__ = [
    "T5Gemma2TextScaledWordEmbedding",
    "T5Gemma2MultiModalProjector",
    "T5Gemma2RotaryEmbedding",
    "T5Gemma2MLP",
    "T5Gemma2SelfAttention",
    "T5Gemma2EncoderLayer",
    "T5Gemma2Encoder",
    "_t5gemma2_use_optimized_paths",
    "apply_rotary_pos_emb",
    "repeat_kv",
    "eager_attention_forward",
]
