"""Stable pooler protocol — zero vLLM imports.

Defines the contract between business pooling logic (GLiNER, ColBERT, etc.)
and the vLLM adapter layer.  When vLLM changes its Pooler ABC or
PoolingMetadata layout, only ``vllm_adapter.py`` needs updating — everything
that implements ``FactoryPooler`` stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class PoolerContext:
    """Version-independent per-batch metadata that business poolers receive.

    The adapter translates vLLM's ``PoolingMetadata`` into this before
    calling ``FactoryPooler.forward()``.
    """

    seq_lengths: list[int]
    extra_kwargs: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    prompt_token_ids: list[torch.Tensor] = field(default_factory=list)
    device: torch.device | None = None
    dtype: torch.dtype | None = None


def split_hidden_states(
    hidden_states: torch.Tensor,
    seq_lengths: list[int],
) -> list[torch.Tensor]:
    """Split concatenated hidden states into per-sequence tensors.

    Pure torch, no vLLM dependency.  Works on both CPU and GPU.
    """
    parts: list[torch.Tensor] = []
    offset = 0
    for length in seq_lengths:
        parts.append(hidden_states[offset : offset + length])
        offset += length
    return parts


@runtime_checkable
class FactoryPooler(Protocol):
    """Stable interface that business pooling logic implements.

    Implementations must never import from ``vllm.*``.
    """

    def get_tasks(self) -> set[str]:
        """Return the set of pooling task names this pooler supports."""
        ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        """Run business pooling logic.

        Args:
            hidden_states: Concatenated hidden states for all sequences
                in the batch, shape ``(total_tokens, hidden_dim)``.
            ctx: Batch metadata (sequence lengths, extra kwargs, etc.).

        Returns:
            One tensor per sequence (or ``None`` for unfinished sequences
            when chunked prefill is active).
        """
        ...


class PassthroughPooler:
    """No-op pooler for models that handle everything in ``forward()``.

    Splits concatenated hidden states per sequence and returns them unchanged.
    When wrapped by ``VllmPoolerAdapter``, the adapter delegates to vLLM's
    ``AllPool`` for GPU-efficient splitting with chunked-prefill support,
    so ``forward()`` here serves only as a pure-torch fallback.
    """

    def get_tasks(self) -> set[str]:
        return {"token_embed"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        return split_hidden_states(hidden_states, ctx.seq_lengths)
