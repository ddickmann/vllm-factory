"""
NemotronColEmbed — vLLM Qwen3-VL with bidirectional attention + L2-normalized embeddings.

Key differences from ColQwen3:
    1. No projection layer — outputs raw hidden_size (2560) embeddings
    2. Bidirectional attention (is_causal=False on all attention layers)
    3. Skips final RMSNorm — uses pre-norm last-layer output
    4. L2 normalization on masked hidden states

Backbone: vLLM's Qwen3VLForConditionalGeneration (built-in, fully optimized)
Weights:  Standard Qwen3VL prefix mapping (identical to base model)
"""

import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_factory.pooling.protocol import PassthroughPooler
from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

logger = logging.getLogger(__name__)


def _patch_encoder_only_kv_cache_spec() -> None:
    """Patch ``Attention.get_kv_cache_spec`` to return None for encoder-only layers.

    WHY: vLLM 0.19 allocates KV cache for all attention layers by default.
    Encoder-only models with ``AttentionType.ENCODER_ONLY`` (bidirectional
    attention, no autoregressive decoding) do not need KV cache. Without this
    patch, vLLM reserves unnecessary GPU memory and may raise shape mismatches.

    CHARACTERISTICS: Idempotent (guarded by ``_vllm_factory_nemotron_patch``
    attribute). Applied in ``NemotronColEmbedModel.__init__``, not at import
    time. Only modifies the return value for ENCODER_ONLY attention layers.

    UPSTREAM RESOLUTION: Remove once vLLM natively returns ``None`` from
    ``get_kv_cache_spec`` for ``AttentionType.ENCODER_ONLY`` layers, or
    provides a model-level opt-out for KV cache allocation on encoder models.
    """
    try:
        from vllm.attention.layer import Attention
        from vllm.v1.attention.backend import AttentionType
    except ImportError:
        logger.debug("Could not patch Attention.get_kv_cache_spec (vLLM API change)")
        return

    current = Attention.get_kv_cache_spec
    if getattr(current, "_vllm_factory_nemotron_patch", False):
        return

    def _patched(self, vllm_config):
        if self.attn_type == AttentionType.ENCODER_ONLY:
            return None
        return current(self, vllm_config)

    _patched._vllm_factory_nemotron_patch = True  # type: ignore[attr-defined]
    Attention.get_kv_cache_spec = _patched


@default_pooling_type(tok_pooling_type="ALL")
class NemotronColEmbedModel(Qwen3VLForConditionalGeneration):
    """Nemotron ColEmbed: bidirectional Qwen3-VL for visual document retrieval.

    Outputs (num_tokens, hidden_size) L2-normalized embeddings per input
    for late-interaction ColBERT MaxSim scoring.
    """

    is_pooling_model = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        _patch_encoder_only_kv_cache_spec()

        # CRITICAL: Patch is_causal=False BEFORE super().__init__() so that
        # Qwen3DecoderLayer sees it and uses AttentionType.ENCODER_ONLY
        # for bidirectional attention (vLLM's qwen3.py:176-179).
        config = vllm_config.model_config.hf_config
        config.is_causal = False
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            text_config.is_causal = False

        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Replace final RMSNorm with identity (skip-norm).
        # Qwen2Model.forward() calls self.norm(hidden_states, residual) which
        # applies RMSNorm + adds residual. Nemotron skips the norm and just adds
        # the residual. SkipNorm mimics the RMSNorm interface but only adds residual.
        class SkipNorm(nn.Module):
            """Drop-in replacement for vLLM RMSNorm that skips normalization."""

            def forward(self, x, residual=None):
                if residual is not None:
                    return x + residual, residual
                return x, residual

        self.language_model.model.norm = SkipNorm()

        config = vllm_config.model_config.hf_config
        hidden_size = getattr(getattr(config, "text_config", config), "hidden_size", 2560)

        # Pooler for token-level embeddings (ALL tokens)
        self.pooler = VllmPoolerAdapter(
            PassthroughPooler(),
            pooler_config=vllm_config.model_config.pooler_config,
        )

        logger.info(
            f"[NemotronColEmbed] Initialized: hidden_size={hidden_size}, "
            f"pooling=ALL, no projection, skip final norm, L2 norm on output"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: Qwen3VL encoder → L2-normalized hidden states.

        The parent Qwen3VLForConditionalGeneration.forward() calls
        self.language_model.model() which applies the final RMSNorm.
        Nemotron-colembed skips the final norm in the HF reference, but
        we keep it here since vLLM's Qwen3LLMModel always applies it.
        The L2 normalization then handles the magnitude normalization.
        """
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # L2 normalize per token (critical for ColBERT MaxSim)
        hidden_states = F.normalize(hidden_states, p=2, dim=-1)

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(iter(list(weights)), mapper=self.hf_to_vllm_mapper)
        # Mark constructor-initialized params as loaded for vLLM 0.19+ validation
        for name in dict(self.named_parameters()):
            loaded.add(name)
        return loaded
