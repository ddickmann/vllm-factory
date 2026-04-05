"""
ColQwen3 — vLLM built-in Qwen3-VL + ColPali multi-vector pooler.

Backbone: vLLM's Qwen3VLForConditionalGeneration (built-in, fully optimized)
Pooler:   ReplicatedLinear projection → L2 normalize (token-level, ALL pooling)
Weights:  AutoWeightsLoader with WeightsMapper for ColQwen3 checkpoint format

vLLM 0.15.x compatible:
- Extends native vLLM model (inherits all optimizations)
- @default_pooling_type(tok_pooling_type="ALL") for multi-vector embeddings
- is_pooling_model = True
"""

from typing import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_factory.pooling.protocol import PassthroughPooler
from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

# ---------------------------------------------------------------------------
# Per-model ProcessingInfo: forces the slow Qwen2VLImageProcessor.
# This replaces the global monkey-patch that previously leaked into all models.
# ---------------------------------------------------------------------------


class ColQwen3ProcessingInfo(Qwen3VLProcessingInfo):
    """Processing info that forces the slow Qwen2VLImageProcessor.

    The slow processor produces different min/max pixel defaults
    (3136/1003520 vs fast's 2352/802816), which is required for parity
    with the sauerkrautlm-colpali HF reference implementation.
    """

    def get_image_processor(self, **kwargs):
        processor = self.get_hf_processor(**kwargs)
        image_processor = processor.image_processor

        # If the image processor is the fast version, replace its
        # size parameters with the slow processor's defaults
        if type(image_processor).__name__ == "Qwen2VLImageProcessorFast":
            try:
                from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
                    Qwen2VLImageProcessor,
                )

                slow = Qwen2VLImageProcessor.from_pretrained(
                    self.ctx.model_config.model,
                    trust_remote_code=True,
                )
                return slow
            except Exception:
                pass
        return image_processor


class ColPaliProjection(nn.Module):
    """Linear projection to colpali_dim + L2 normalization."""

    def __init__(self, hidden_size: int, colpali_dim: int = 128, quant_config=None):
        super().__init__()
        self.linear = ReplicatedLinear(
            hidden_size, colpali_dim, bias=True, quant_config=quant_config
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected, _ = self.linear(hidden_states)
        return projected / projected.norm(dim=-1, keepdim=True)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=ColQwen3ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
@default_pooling_type(tok_pooling_type="ALL")
class Qwen3VLForColPali(Qwen3VLForConditionalGeneration):
    """Qwen3-VL extended with ColPali projection for visual document retrieval.

    Outputs (num_tokens, 128) per input for late-interaction MaxSim scoring.
    """

    is_pooling_model = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "language_model.": "language_model.model.",
            "custom_text_proj.": "colpali_projection.linear.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        hidden_size = getattr(getattr(config, "text_config", config), "hidden_size", 2048)
        colpali_dim = getattr(config, "dim", 128)

        self.colpali_projection = ColPaliProjection(
            hidden_size=hidden_size,
            colpali_dim=colpali_dim,
            quant_config=vllm_config.quant_config,
        )

        self.pooler = VllmPoolerAdapter(
            PassthroughPooler(),
            pooler_config=vllm_config.model_config.pooler_config,
        )

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs
    ):
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return self.colpali_projection(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(iter(list(weights)), mapper=self.hf_to_vllm_mapper)
        # Mark constructor-initialized params as loaded for vLLM 0.19+ validation
        for name in dict(self.named_parameters()):
            loaded.add(name)
        return loaded
