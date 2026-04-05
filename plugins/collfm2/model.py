"""
ColLFM2 Model for vLLM 0.14.0 - LFM2-VL with ColPali Pooling

This module extends vLLM's native Lfm2VLForConditionalGeneration to add
ColPali-style multi-vector embedding support for the smaller LFM2-VL backbone.

The model:
1. Uses vLLM's optimized LFM2-VL vision-language encoder
2. Adds a linear projection layer (hidden_size -> 128)
3. L2 normalizes per-token embeddings
4. Returns multi-vector embeddings for late interaction (ALL pooling)

Key Features:
- Extends vLLM's native LFM2-VL (fully optimized for vLLM 0.14.0)
- Loads projection weights from custom_text_proj in checkpoint
- is_pooling_model = True with default_pooling_type = "ALL"
- Custom weight mapper for ColLFM2 checkpoint format
- Much smaller (~0.9GB) compared to ColQwen3 (~5GB+)
"""

import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm_factory.pooling.protocol import PassthroughPooler
from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter
from vllm.model_executor.models.interfaces_base import default_pooling_type

# Import the native vLLM LFM2-VL
from vllm.model_executor.models.lfm2_vl import Lfm2VLForConditionalGeneration
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

logger = logging.getLogger(__name__)


class ColLFM2Projection(nn.Module):
    """ColLFM2 projection: linear projection to 128-dim + L2 normalization.

    This is the equivalent of ColLFM2's custom_text_proj layer.
    Applied after hidden_states are pooled (ALL tokens).
    """

    def __init__(
        self,
        hidden_size: int,
        colpali_dim: int = 128,
        quant_config=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.colpali_dim = colpali_dim

        # Linear projection with bias (matching ColLFM2's custom_text_proj)
        self.linear = ReplicatedLinear(
            hidden_size,
            colpali_dim,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project and normalize embeddings.

        Args:
            hidden_states: (seq_len, hidden_size) or (batch, seq_len, hidden_size)

        Returns:
            Projected and normalized embeddings (seq_len, colpali_dim)
        """
        # Project to ColPali dimension
        projected, _ = self.linear(hidden_states)

        # L2 normalize per token (critical for MaxSim) - NO epsilon like original!
        projected = projected / projected.norm(dim=-1, keepdim=True)

        return projected


@default_pooling_type(tok_pooling_type="ALL")  # Return all token embeddings for late interaction
class LFM2VLForColPali(Lfm2VLForConditionalGeneration):
    """LFM2-VL with ColPali pooling for visual document retrieval.

    Extends vLLM's native Lfm2VLForConditionalGeneration to add:
    1. is_pooling_model = True for embedding task
    2. ColPali projection layer (hidden_size -> 128 + L2 norm)
    3. Uses "ALL" pooling type for multi-vector embeddings
    4. vLLM Pooler for token embedding

    The model outputs (num_tokens, 128) embeddings per input,
    suitable for late interaction retrieval with MaxSim.

    Key advantages over ColQwen3:
    - Much smaller: ~0.9GB vs ~5GB+
    - Faster inference due to smaller model
    - Still achieves #1 benchmark scores for small models
    """

    # Tell vLLM this is a pooling model
    is_pooling_model = True

    # Custom weight mapper for ColLFM2 checkpoint format
    # ColLFM2 uses: model.*, custom_text_proj.* (may be prefixed with model.)
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # From parent Lfm2VLForConditionalGeneration:
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            # ColLFM2 specific - map custom_text_proj to collfm2_projection.linear
            # Handle both "model.custom_text_proj.*" and "custom_text_proj.*"
            "model.custom_text_proj.": "collfm2_projection.linear.",
            "custom_text_proj.": "collfm2_projection.linear.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        # Get hidden size from config (LFM2-VL uses text_config.hidden_size)
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            hidden_size = config.text_config.hidden_size
        else:
            hidden_size = getattr(config, "hidden_size", 512)  # LFM2-VL-450M default

        # Get ColPali dim from config (default 128)
        colpali_dim = getattr(config, "dim", 128)

        # ColLFM2 projection layer
        self.collfm2_projection = ColLFM2Projection(
            hidden_size=hidden_size,
            colpali_dim=colpali_dim,
            quant_config=quant_config,
        )

        # Create vLLM pooler for token embedding task
        # This handles splitting by sequence and output format
        self.pooler = VllmPoolerAdapter(
            PassthroughPooler(),
            pooler_config=vllm_config.model_config.pooler_config,
        )

        # Store model path for weight loading
        self.model_path = vllm_config.model_config.model

        logger.info("[LFM2VLForColPali] Initialized with:")
        logger.info("  Hidden size: %s", hidden_size)
        logger.info("  ColPali dim: %s", colpali_dim)
        logger.info("  Model path: %s", self.model_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass - get hidden states and apply ColLFM2 projection.

        For pooling models, we return the projected hidden states.
        vLLM's pooler (with "ALL" type) will split them by sequence.
        """
        # Get hidden states from parent (LFM2-VL)
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # Apply ColLFM2 projection (linear + L2 normalize)
        projected = self.collfm2_projection(hidden_states)

        return projected

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with custom mapping for ColLFM2 format.

        Handles ColLFM2's checkpoint format which has:
        - model.* (standard LFM2-VL weights)
        - custom_text_proj.* (map to collfm2_projection.linear.*)
        """
        # Convert to list to allow multiple passes
        weights_list = list(weights)

        # Debug: check for custom_text_proj weights
        proj_weights = [(n, w) for n, w in weights_list if "custom_text_proj" in n]
        if proj_weights:
            logger.info(
                "[LFM2VLForColPali] Found projection weights: %s",
                [n for n, _ in proj_weights],
            )

        # Use AutoWeightsLoader with our custom mapper
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(iter(weights_list), mapper=self.hf_to_vllm_mapper)

        # Verify projection weights are loaded
        if hasattr(self, "collfm2_projection"):
            proj_weight = self.collfm2_projection.linear.weight
            # Convert to float for check (FP8 doesn't support .abs() directly)
            try:
                weight_sum = proj_weight.float().abs().sum().item()
                if weight_sum > 0:
                    logger.info(
                        "[LFM2VLForColPali] Projection weights loaded: %s",
                        proj_weight.shape,
                    )
                else:
                    logger.warning(
                        "[LFM2VLForColPali] Projection weights may be zero!"
                    )
            except Exception as e:
                logger.warning(
                    "[LFM2VLForColPali] Could not verify projection weights: %s",
                    e,
                )

        # Mark constructor-initialized params as loaded for vLLM 0.19+ validation
        for name in dict(self.named_parameters()):
            loaded.add(name)
        return loaded
