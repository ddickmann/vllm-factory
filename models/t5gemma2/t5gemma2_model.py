"""T5Gemma2 decoder and seq2seq model for vLLM Factory."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from kernels.fused_gemma_rms_norm_dropout_residual import (
    fused_gemma_rms_norm_dropout_residual,
)
from kernels.fused_rope_global import fused_rope_global_apply
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.utils import make_layers
from vllm.sequence import IntermediateTensors

from .config import (
    T5Gemma2Config,
    T5Gemma2DecoderConfig,
    get_t5gemma2_text_config,
)
from .t5gemma2_encoder import (
    T5Gemma2Encoder,
    T5Gemma2MLP,
    T5Gemma2RotaryEmbedding,
    T5Gemma2TextScaledWordEmbedding,
    _build_segment_ids,
    _resolve_model_config,
    _t5gemma2_use_optimized_paths,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)


def _build_decoder_attention_masks(
    decoder_attention_mask: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    sliding_window: int | None,
) -> dict[str, torch.Tensor]:
    """Build merged self+cross masks for decoder full and sliding attention."""

    decoder_attention_mask = decoder_attention_mask.to(dtype=torch.bool)
    encoder_attention_mask = encoder_attention_mask.to(dtype=torch.bool)

    decoder_pair_mask = decoder_attention_mask[:, :, None] & decoder_attention_mask[:, None, :]
    causal_mask = torch.tril(
        torch.ones(
            decoder_attention_mask.shape[1],
            decoder_attention_mask.shape[1],
            device=decoder_attention_mask.device,
            dtype=torch.bool,
        )
    )
    segment_ids = _build_segment_ids(position_ids)
    block_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
    full_self_mask = decoder_pair_mask & causal_mask[None, :, :] & block_mask

    cross_mask = decoder_attention_mask[:, :, None] & encoder_attention_mask[:, None, :]

    merged_masks = {
        "full_attention": torch.cat(
            [full_self_mask[:, None, :, :], cross_mask[:, None, :, :]],
            dim=-1,
        )
    }

    if sliding_window is not None:
        seq_len = decoder_attention_mask.shape[1]
        rows = torch.arange(seq_len, device=decoder_attention_mask.device)
        dist = rows[:, None] - rows[None, :]
        sliding_mask = (dist >= 0) & (dist < sliding_window)
        merged_masks["sliding_attention"] = torch.cat(
            [
                (full_self_mask & sliding_mask[None, :, :])[:, None, :, :],
                cross_mask[:, None, :, :],
            ],
            dim=-1,
        )
    else:
        merged_masks["sliding_attention"] = merged_masks["full_attention"]

    return merged_masks


class T5Gemma2MergedAttention(nn.Module):
    """HF-exact merged self+cross attention for the decoder."""

    def __init__(
        self,
        config: T5Gemma2DecoderConfig,
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
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_kv_heads
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

    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        merged_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        encoder_seq_len = encoder_hidden_states.shape[1]

        q, _ = self.q_proj(hidden_states)
        k_self, _ = self.k_proj(hidden_states)
        v_self, _ = self.v_proj(hidden_states)
        k_cross, _ = self.k_proj(encoder_hidden_states)
        v_cross, _ = self.v_proj(encoder_hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_self = k_self.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_self = v_self.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k_cross = k_cross.view(batch_size, encoder_seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_cross = v_cross.view(batch_size, encoder_seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k_self = self.k_norm(k_self)
        k_cross = self.k_norm(k_cross)

        q, k_self = apply_rotary_pos_emb(q, k_self, *position_embeddings)
        k = torch.cat([k_self, k_cross], dim=2)
        v = torch.cat([v_self, v_cross], dim=2)

        k = repeat_kv(k, self.num_queries_per_kv)
        v = repeat_kv(v, self.num_queries_per_kv)

        if self.attn_softcap is None:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=merged_attention_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
                scale=self.scaling,
            )
        else:
            attn_output = eager_attention_forward(
                q,
                k,
                v,
                attention_mask=merged_attention_mask,
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
        encoder_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        merged_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (not _t5gemma2_use_optimized_paths()) or self.attn_softcap is not None:
            return self._forward_reference(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                position_embeddings=position_embeddings,
                merged_attention_mask=merged_attention_mask,
            )

        batch_size, seq_len, _ = hidden_states.shape
        encoder_seq_len = encoder_hidden_states.shape[1]

        q, _ = self.q_proj(hidden_states)
        k_self, _ = self.k_proj(hidden_states)
        v_self, _ = self.v_proj(hidden_states)
        k_cross, _ = self.k_proj(encoder_hidden_states)
        v_cross, _ = self.v_proj(encoder_hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_self = k_self.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v_self = v_self.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k_cross = k_cross.view(batch_size, encoder_seq_len, self.num_kv_heads, self.head_dim)
        v_cross = v_cross.view(batch_size, encoder_seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k_self = self.k_norm(k_self)
        k_cross = self.k_norm(k_cross)

        q, k_self = fused_rope_global_apply(q, k_self, *position_embeddings)
        q = q.transpose(1, 2)
        k_self = k_self.transpose(1, 2)
        v_self = v_self.transpose(1, 2)
        k_cross = k_cross.transpose(1, 2)
        v_cross = v_cross.transpose(1, 2)

        k = torch.cat([k_self, k_cross], dim=2)
        v = torch.cat([v_self, v_cross], dim=2)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=merged_attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            scale=self.scaling,
            enable_gqa=self.num_heads > self.num_kv_heads,
        )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        attn_output, _ = self.o_proj(attn_output)
        return attn_output


class T5Gemma2DecoderLayer(nn.Module):
    """One T5Gemma2 decoder block with merged attention."""

    def __init__(
        self,
        config: T5Gemma2DecoderConfig,
        *,
        layer_idx: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = T5Gemma2MergedAttention(
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
        encoder_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        merged_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            merged_attention_mask=merged_attention_mask,
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


class T5Gemma2Decoder(nn.Module):
    """HF-exact text decoder for T5Gemma2."""

    def __init__(
        self,
        config_or_vllm_config: VllmConfig | T5Gemma2Config | T5Gemma2DecoderConfig,
        *,
        prefix: str = "decoder",
    ) -> None:
        super().__init__()
        outer_config, text_config, cache_config, quant_config = _resolve_model_config(
            config_or_vllm_config,
            is_encoder=False,
        )
        del cache_config  # This implementation focuses on full-sequence parity.
        self.outer_config = outer_config
        self.config = text_config
        self.quant_config = quant_config
        eoi_token_index = getattr(outer_config, "eoi_token_index", None)

        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            embed_scale=text_config.hidden_size**0.5,
            eoi_token_index=eoi_token_index,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.dropout = nn.Dropout(getattr(text_config, "dropout_rate", 0.0))
        self.rotary_emb = T5Gemma2RotaryEmbedding(text_config)
        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config.num_hidden_layers,
            lambda prefix: T5Gemma2DecoderLayer(
                text_config,
                layer_idx=int(prefix.split(".")[-1]),
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = GemmaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def get_input_embeddings(self) -> T5Gemma2TextScaledWordEmbedding:
        return self.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if inputs_embeds is not None and inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
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

        return input_ids, attention_mask, position_ids, inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        **_: object,
    ) -> torch.Tensor | IntermediateTensors:
        if encoder_hidden_states is None:
            raise ValueError("`encoder_hidden_states` must be provided to the decoder.")

        if encoder_hidden_states.dim() == 2:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
        if encoder_attention_mask is not None and encoder_attention_mask.dim() == 1:
            encoder_attention_mask = encoder_attention_mask.unsqueeze(0)

        input_ids, attention_mask, position_ids, inputs_embeds = self._prepare_inputs(
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
        )
        batch_size, decoder_seq_len = attention_mask.shape
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                device=encoder_hidden_states.device,
                dtype=torch.bool,
            )

        if get_pp_group().is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids)
                if inputs_embeds is None
                else inputs_embeds
            )
            hidden_states = self.dropout(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        del batch_size, decoder_seq_len
        merged_masks = _build_decoder_attention_masks(
            attention_mask,
            encoder_attention_mask,
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
                encoder_hidden_states=encoder_hidden_states,
                position_embeddings=position_embeddings[layer_type],
                merged_attention_mask=merged_masks[layer_type],
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        for name, loaded_weight in weights:
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

        return loaded_params


class T5Gemma2Model(nn.Module):
    """T5Gemma2 encoder-decoder model."""

    def __init__(self, config_or_vllm_config: VllmConfig | T5Gemma2Config, *, prefix: str = "model") -> None:
        super().__init__()
        self.encoder = T5Gemma2Encoder(
            config_or_vllm_config,
            prefix=f"{prefix}.encoder",
        )
        self.decoder = T5Gemma2Decoder(config_or_vllm_config, prefix=f"{prefix}.decoder")

    def get_encoder_outputs(
        self,
        input_ids: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if input_ids is None and inputs_embeds is None:
            return None
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        decoder_position_ids: torch.Tensor | None = None,
        encoder_outputs: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        decoder_inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_outputs is None:
            encoder_outputs = self.get_encoder_outputs(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
            )

        return self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stripped = [
            (name[len("model.") :] if name.startswith("model.") else name, weight)
            for name, weight in weights
        ]
        encoder_weights = [
            (name[len("encoder.") :], weight)
            for name, weight in stripped
            if name.startswith("encoder.")
        ]
        decoder_weights = [
            (name[len("decoder.") :], weight)
            for name, weight in stripped
            if name.startswith("decoder.")
        ]

        loaded_params: set[str] = set()
        for name in self.encoder.load_weights(encoder_weights):
            loaded_params.add(f"encoder.{name}")
        for name in self.decoder.load_weights(decoder_weights):
            loaded_params.add(f"decoder.{name}")
        return loaded_params


class T5Gemma2ForConditionalGeneration(nn.Module, SupportsLoRA):
    """Conditional generation wrapper with logits processing."""

    def __init__(self, config_or_vllm_config: VllmConfig | T5Gemma2Config, *, prefix: str = "") -> None:
        super().__init__()
        outer_config, _, _, _ = _resolve_model_config(config_or_vllm_config, is_encoder=False)
        self.config = outer_config
        self.model = T5Gemma2Model(config_or_vllm_config, prefix=f"{prefix}.model" if prefix else "model")
        decoder_config = get_t5gemma2_text_config(outer_config, is_encoder=False)
        self.logits_processor = LogitsProcessor(
            decoder_config.vocab_size,
            soft_cap=decoder_config.final_logit_softcapping,
        )

    def get_language_model(self) -> nn.Module:
        return self.model.decoder

    def get_encoder_outputs(
        self,
        input_ids: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        return self.model.get_encoder_outputs(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.decoder.embed_input_ids(input_ids)

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        decoder_position_ids: torch.Tensor | None = None,
        encoder_outputs: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        decoder_inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            pixel_values=pixel_values,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.model.encoder.embed_tokens, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str] | None:
        weights_list = list(weights)
        model_weights = [
            (name[len("model.") :] if name.startswith("model.") else name, weight)
            for name, weight in weights_list
            if not name.startswith("lm_head.")
        ]
        self.model.load_weights(model_weights)
        return None


__all__ = [
    "T5Gemma2MergedAttention",
    "T5Gemma2DecoderLayer",
    "T5Gemma2Decoder",
    "T5Gemma2Model",
    "T5Gemma2ForConditionalGeneration",
]
