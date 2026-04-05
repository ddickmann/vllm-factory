"""Pooler registry — maps pooler names to FactoryPooler implementations.

Ships built-in poolers (mean, cls, normalized_mean, passthrough) and provides
a public ``register_pooler()`` API for users to add custom poolers at runtime.

Pooler classes must implement the ``FactoryPooler`` protocol::

    class MyPooler:
        def __init__(self, hidden_size: int = 768, **kwargs): ...
        def get_tasks(self) -> set[str]: ...
        def forward(self, hidden_states: Tensor, ctx: PoolerContext) -> list[Tensor | None]: ...

They are instantiated at model construction time with
``(hidden_size=hidden_size, **pooler_config)``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register_pooler(name: str, cls: type) -> None:
    """Register a pooler class under *name*.

    The class must implement the ``FactoryPooler`` protocol (``get_tasks``
    and ``forward``).  It will be instantiated with ``(hidden_size, **pooler_config)``
    at model construction time.
    """
    _REGISTRY[name] = cls
    logger.debug("Registered pooler '%s' -> %s", name, cls.__name__)


def get_pooler_cls(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown pooler '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_poolers() -> list[str]:
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in poolers
# ---------------------------------------------------------------------------


class MeanPooler:
    """Average all token embeddings per sequence."""

    def __init__(self, hidden_size: int = 0, **kwargs: Any) -> None:
        pass

    def get_tasks(self) -> set[str]:
        return {"embed", "plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        parts = split_hidden_states(hidden_states, ctx.seq_lengths)
        return [p.mean(dim=0) for p in parts]


class CLSPooler:
    """Take the first token ([CLS]) embedding per sequence."""

    def __init__(self, hidden_size: int = 0, **kwargs: Any) -> None:
        pass

    def get_tasks(self) -> set[str]:
        return {"embed", "plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        parts = split_hidden_states(hidden_states, ctx.seq_lengths)
        return [p[0] for p in parts]


class NormalizedMeanPooler:
    """Mean pooling followed by L2 normalization."""

    def __init__(self, hidden_size: int = 0, **kwargs: Any) -> None:
        pass

    def get_tasks(self) -> set[str]:
        return {"embed", "plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        parts = split_hidden_states(hidden_states, ctx.seq_lengths)
        return [F.normalize(p.mean(dim=0), p=2, dim=-1) for p in parts]


# ---------------------------------------------------------------------------
# Register builtins + wrappers for shared poolers in poolers/
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    register_pooler("mean", MeanPooler)
    register_pooler("cls", CLSPooler)
    register_pooler("normalized_mean", NormalizedMeanPooler)

    try:
        from poolers.colbert import ColBERTPooler

        register_pooler("colbert", ColBERTPooler)
    except ImportError:
        logger.debug("ColBERTPooler not available")

    try:
        from poolers.colpali import ColPaliPooler

        register_pooler("colpali", ColPaliPooler)
    except ImportError:
        logger.debug("ColPaliPooler not available")

    try:
        from poolers.gliner import GLiNERSpanPooler

        register_pooler("gliner", GLiNERSpanPooler)
    except ImportError:
        logger.debug("GLiNERSpanPooler not available")

    try:
        from vllm_factory.pooling.protocol import PassthroughPooler

        register_pooler("passthrough", PassthroughPooler)
    except ImportError:
        logger.debug("PassthroughPooler not available")


_register_builtins()
