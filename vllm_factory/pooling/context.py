"""Stable pooler context — decouples pooler math from vLLM metadata layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class SequenceContext:
    """Per-sequence metadata extracted from vLLM pooling metadata."""
    hidden_states: torch.Tensor
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    seq_len: int = 0
    token_ids: Optional[list[int]] = None


@dataclass
class PoolerContext:
    """Stable interface that pooler kernels receive instead of raw PoolingMetadata."""
    sequences: list[SequenceContext] = field(default_factory=list)
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None


@dataclass
class PoolerResult:
    """What pooler kernels return — a list of per-sequence outputs."""
    outputs: list[torch.Tensor] = field(default_factory=list)
