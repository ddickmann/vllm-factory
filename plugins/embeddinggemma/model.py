"""EmbeddingGemma model for vLLM — uses HF's Gemma3TextModel backbone.

Uses HuggingFace's actual model to guarantee numerical parity with
the SentenceTransformers reference. Pooling uses vLLM's native
DispatchPooler which auto-loads the ST Dense projection layers:

  MEAN pooling → Dense1 (768→3072) → Dense2 (3072→768) → L2 normalize
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces_base import default_pooling_type

from .config import EmbeddingGemmaConfig


@default_pooling_type(seq_pooling_type="MEAN")
class EmbeddingGemmaModel(nn.Module):
    """Gemma3 embedding model using HF backbone + vLLM DispatchPooler.

    Pipeline mirrors SentenceTransformers exactly:
      HF Gemma3TextModel → MEAN pool → Dense1 → Dense2 → L2 normalize

    Since is_pooling_model=True, vLLM's as_embedding_model() adapter
    skips wrapping and expects self.pooler to exist. We create it using
    vLLM's DispatchPooler.for_embedding() which auto-loads the Dense
    layers from the HF repo and applies L2 normalization.

    NOTE: Must be run with dtype=float32. Gemma's embedding scale
    (sqrt(hidden_size) ≈ 27.7) overflows float16 range, producing NaN.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: EmbeddingGemmaConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config

        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(
            config._name_or_path,
            config=config,
            trust_remote_code=True,
        )
        self.backbone.eval()

        from vllm.model_executor.layers.pooler import DispatchPooler
        pooler_config = vllm_config.model_config.pooler_config
        self.pooler = DispatchPooler.for_embedding(pooler_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        """Run HF backbone and return hidden states for pooler."""
        position_ids = positions.unsqueeze(0)
        input_ids_2d = input_ids.unsqueeze(0)

        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids_2d,
                position_ids=position_ids,
            )

        return outputs.last_hidden_state.squeeze(0)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Backbone loaded via from_pretrained; consume weight iterator."""
        for _, _ in weights:
            pass
        return set(name for name, _ in self.named_parameters())
