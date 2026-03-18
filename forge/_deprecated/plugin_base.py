"""
Base class for vLLM plugin models.

Provides a lightweight abstract base that encapsulates the common
patterns across all vLLM custom model plugins:
- encoder_only attention type
- weight loading with mapping
- embed_input_ids for pooling runner
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from vllm.sequence import IntermediateTensors


class BasePluginModel(nn.Module):
    """Abstract base class for vLLM custom model plugins.

    This provides the skeleton that all plugin models should follow.
    Subclass and implement:
    - __init__(): set up encoder + pooler + any projection layers
    - forward(): run encoder, apply projection/pooler
    - load_weights(): map checkpoint keys to model parameters

    The @attn_type and @default_pooling_type decorators should be
    applied to your concrete subclass, not inherited from here.

    Example:
        @attn_type("encoder_only")
        @default_pooling_type(tok_pooling_type="ALL")
        class MyModel(BasePluginModel):
            is_pooling_model = True

            def __init__(self, *, vllm_config, prefix=""):
                super().__init__()
                # Build your model here

            def forward(self, input_ids, positions, ...):
                # Run encoder + projection

            def load_weights(self, weights):
                # Load checkpoint weights
    """

    is_pooling_model: bool = True

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the model.

        vLLM pooling models receive:
        - input_ids: (total_tokens,) — 1D tensor with all sequences concatenated
        - positions: (total_tokens,) — 1D position IDs

        Returns:
            Processed tensor for the pooler
        """
        ...

    @abstractmethod
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from a checkpoint.

        Must handle key mapping from the checkpoint format to the
        model's parameter names.

        Returns:
            Set of parameter names that were loaded
        """
        ...

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from input_ids.

        Required by vLLM 0.14's pooling runner. Override this if your
        embedding layer has a non-standard name.

        Default implementation assumes self.model.embeddings.tok_embeddings
        """
        if hasattr(self, "model") and hasattr(self.model, "embeddings"):
            embeddings = self.model.embeddings
            if hasattr(embeddings, "tok_embeddings"):
                return embeddings.tok_embeddings(input_ids)
            if hasattr(embeddings, "word_embeddings"):
                return embeddings.word_embeddings(input_ids)
        raise NotImplementedError(
            "Override embed_input_ids() or ensure self.model.embeddings.tok_embeddings exists"
        )
