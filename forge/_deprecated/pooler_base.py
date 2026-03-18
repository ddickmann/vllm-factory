# ruff: noqa: F821
"""
Base classes for pooler heads.

Provides abstract base classes for the three main pooler patterns
found in custom vLLM models:

- TokenLevelPooler: Multi-vector output (ColBERT, ColPali)
- SpanPooler: Span extraction (GLiNER)
- CLSPooler: Single-vector CLS embedding
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePooler(nn.Module, ABC):
    """Abstract base for vLLM pooler heads.

    Subclass this to create a custom pooler for your model.
    The pooler receives hidden states from the encoder and produces
    the final output (embeddings, logits, spans, etc.).

    vLLM 0.14.x calls:
        1. get_supported_tasks() — what pooling tasks this supports
        2. forward(hidden_states, pooling_metadata) — produce output
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: "PoolingMetadata",
    ) -> list[torch.Tensor]:
        """Process hidden states into final output.

        Args:
            hidden_states: (total_tokens, hidden_size) — all sequences concatenated
            pooling_metadata: vLLM metadata with sequence info

        Returns:
            List of output tensors, one per sequence
        """
        ...

    def get_prompt_lens(
        self, hidden_states: torch.Tensor, pooling_metadata: "PoolingMetadata"
    ) -> list[int]:
        """Extract per-sequence lengths from pooling metadata."""
        if hasattr(pooling_metadata, "prompt_lens"):
            lens = pooling_metadata.prompt_lens
            return lens.tolist() if hasattr(lens, "tolist") else list(lens)
        raise RuntimeError("Cannot extract prompt_lens from pooling_metadata")

    def extract_per_sequence(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: "PoolingMetadata",
    ) -> list[torch.Tensor]:
        """Split concatenated hidden states into per-sequence tensors."""
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[0]
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        sequences = []
        offset = 0
        for length in prompt_lens:
            sequences.append(hidden_states[offset : offset + length])
            offset += length
        return sequences

    def get_pooling_params(
        self, pooling_metadata: "PoolingMetadata"
    ) -> list:
        """Extract pooling params list from metadata (v0 and v1 compatible)."""
        if hasattr(pooling_metadata, "pooling_params") and pooling_metadata.pooling_params:
            return list(pooling_metadata.pooling_params)
        if hasattr(pooling_metadata, "seq_groups") and pooling_metadata.seq_groups:
            params = []
            for seq_ids, pooling_params in pooling_metadata.seq_groups:
                for _ in seq_ids:
                    params.append(pooling_params)
            return params
        return []

    @staticmethod
    def get_additional_data(pooling_params) -> dict:
        """Extract additional_data / extra_kwargs from pooling params."""
        if pooling_params is None:
            return {}
        for attr in ("extra_kwargs", "additional_data", "additional_metadata"):
            data = getattr(pooling_params, attr, None)
            if isinstance(data, dict):
                return data
        return {}


class TokenLevelPooler(BasePooler):
    """Base for multi-vector poolers (ColBERT, ColPali).

    Projects each token to an embedding dim and L2-normalizes.
    Returns a matrix of (seq_len, embed_dim) per sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        output_dim: int = 128,
        normalize: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.normalize = normalize

        from vllm.model_executor.layers.linear import ReplicatedLinear
        self.linear = ReplicatedLinear(
            hidden_size, output_dim, bias=bias, quant_config=None
        )

    def project_and_normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project to output dim and L2-normalize."""
        projected, _ = self.linear(embeddings)
        if self.normalize:
            projected = F.normalize(projected, p=2, dim=-1)
        return projected


class SpanPooler(BasePooler):
    """Base for span extraction poolers (GLiNER).

    Takes hidden states and produces span-entity logits using
    LSTM word encoding, span representations, and entity projections.
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: "PoolingMetadata",
    ) -> list[torch.Tensor]:
        """Returns list of (L, max_width, num_classes) logit tensors per sequence."""
        ...


class CLSPooler(BasePooler):
    """Base for single-vector poolers (sentence embeddings).

    Extracts the [CLS] token embedding and optionally projects + normalizes.
    """

    def __init__(
        self,
        hidden_size: int,
        output_dim: int | None = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.normalize = normalize

        if output_dim and output_dim != hidden_size:
            self.projection = nn.Linear(hidden_size, output_dim)
        else:
            self.projection = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: "PoolingMetadata",
    ) -> list[torch.Tensor]:
        """Extract CLS (first token) embedding per sequence."""
        sequences = self.extract_per_sequence(hidden_states, pooling_metadata)
        outputs = []
        for seq in sequences:
            cls_embed = seq[0]  # First token
            if self.projection is not None:
                cls_embed = self.projection(cls_embed)
            if self.normalize:
                cls_embed = F.normalize(cls_embed, p=0, dim=-1)
            outputs.append(cls_embed)
        return outputs
