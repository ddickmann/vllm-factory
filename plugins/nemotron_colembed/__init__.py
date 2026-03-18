"""NemotronColEmbed — Qwen3-VL with bidirectional attention + L2-normalized ColBERT embeddings.

Includes a runtime patch for Attention.get_kv_cache_spec() to handle
ENCODER_ONLY attention types in Qwen3-VL when is_causal=False.
"""

import logging

from forge.registration import register_plugin

from .config import NemotronColEmbedConfig
from .model import NemotronColEmbedModel

logger = logging.getLogger(__name__)


def _patch_encoder_only_kv_cache_spec() -> None:
    """Patch Attention.get_kv_cache_spec to return None for ENCODER_ONLY layers.

    vLLM's Attention.get_kv_cache_spec() asserts attn_type == DECODER,
    but when is_causal=False is set on Qwen3 config, the attention layers
    become ENCODER_ONLY. These layers don't need KV cache.
    """
    try:
        from vllm.attention.layer import Attention
        from vllm.v1.attention.backend import AttentionType

        _original = Attention.get_kv_cache_spec

        def _patched(self, vllm_config):
            if self.attn_type == AttentionType.ENCODER_ONLY:
                return None
            return _original(self, vllm_config)

        Attention.get_kv_cache_spec = _patched
    except ImportError:
        logger.debug("Could not patch Attention.get_kv_cache_spec (vLLM API change)")


def register() -> None:
    _patch_encoder_only_kv_cache_spec()
    register_plugin("qwen3_vl_nemotron_embed", NemotronColEmbedConfig,
                     "NemotronColEmbedModel", NemotronColEmbedModel,
                     aliases=["Qwen3VLNemotronEmbedModel"])


register()
__all__ = ["NemotronColEmbedModel", "NemotronColEmbedConfig"]
