"""EmbeddingGemma model for vLLM — uses HF's Gemma3TextModel backbone.

Uses HuggingFace's actual model to guarantee numerical parity with
the SentenceTransformers reference. Custom EmbeddingGemmaPooler handles
the post-backbone projection pipeline:

  MEAN pooling → Dense1 (768→3072) → Dense2 (3072→768) → L2 normalize

NOTE: Must be run with dtype=float32. Gemma's embedding scale
(sqrt(hidden_size) ~ 27.7) overflows float16 range, producing NaN.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Tuple

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces_base import default_pooling_type

from .config import EmbeddingGemmaConfig
from .pooler import EmbeddingGemmaPooler

logger = logging.getLogger(__name__)


@default_pooling_type(seq_pooling_type="MEAN")
class EmbeddingGemmaModel(nn.Module):
    """Gemma3 embedding model using HF backbone + EmbeddingGemmaPooler.

    Pipeline mirrors SentenceTransformers exactly:
      HF Gemma3TextModel → MEAN pool → Dense1 → Dense2 → L2 normalize

    NOTE: Must be run with dtype=float32. Gemma's embedding scale
    (sqrt(hidden_size) ~ 27.7) overflows float16 range, producing NaN.
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

        from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

        self._business_pooler = EmbeddingGemmaPooler(
            hidden_size=config.hidden_size,
            dtype=torch.float32,
        )
        self.pooler = VllmPoolerAdapter(self._business_pooler)
        self._load_dense_weights(config._name_or_path)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs
    ):
        """Run HF backbone and return hidden states for pooler."""
        position_ids = positions.unsqueeze(0)
        input_ids_2d = input_ids.unsqueeze(0)

        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids_2d,
                position_ids=position_ids,
            )

        return outputs.last_hidden_state.squeeze(0)

    def _load_dense_weights(self, model_name_or_path: str) -> None:
        """Load SentenceTransformers Dense projection weights from HF repo."""
        try:
            from huggingface_hub import hf_hub_download
            import safetensors.torch
        except ImportError:
            logger.warning("huggingface_hub or safetensors not available; "
                           "Dense projection weights not loaded")
            return

        for layer_idx, layer_name, attr in [
            (2, "2_Dense", "dense1"),
            (3, "3_Dense", "dense2"),
        ]:
            try:
                path = hf_hub_download(model_name_or_path,
                                       f"{layer_name}/model.safetensors")
                state = safetensors.torch.load_file(path)
                linear = getattr(self._business_pooler, attr)
                if "linear.weight" in state:
                    linear.weight.data.copy_(state["linear.weight"])
                    logger.info("Loaded %s from %s", attr, layer_name)
            except Exception as exc:
                logger.warning("Could not load %s weights: %s", attr, exc)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Backbone loaded via from_pretrained; consume weight iterator."""
        for _, _ in weights:
            pass
        loaded = set()
        for name in dict(self.named_parameters()):
            loaded.add(name)
        return loaded
