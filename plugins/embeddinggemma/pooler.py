"""EmbeddingGemma Pooler — MEAN pool + Dense1 + Dense2 + L2 normalize.

Mirrors the SentenceTransformers pipeline exactly.

Implements FactoryPooler protocol — zero vLLM imports.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states


class EmbeddingGemmaPooler(nn.Module):
    """Custom pooler for EmbeddingGemma.

    Pipeline: MEAN pooling → Dense1 (768→3072) → Dense2 (3072→768) → L2 normalize
    """

    def __init__(self, hidden_size: int = 768, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.dense1 = nn.Linear(hidden_size, 3072, bias=False, dtype=dtype)
        self.dense2 = nn.Linear(3072, hidden_size, bias=False, dtype=dtype)

    # ── FactoryPooler protocol ───────────────────────────────────────────

    def get_tasks(self) -> set[str]:
        return {"embed", "plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        sequences = split_hidden_states(hidden_states, ctx.seq_lengths)
        outputs: List[torch.Tensor] = []

        for seq_hidden in sequences:
            pooled = seq_hidden.mean(dim=0, keepdim=True)
            projected = self.dense1(pooled)
            projected = self.dense2(projected)
            normalized = F.normalize(projected, p=2, dim=-1)
            outputs.append(normalized.squeeze(0))

        return outputs
