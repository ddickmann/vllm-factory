"""
ColBERT Pooler — Token-level multi-vector embeddings for late interaction retrieval.

Architecture:
    Linear projection: hidden_size (768) → colbert_dim (128)
    L2 normalization per token
    Query padding to fixed length (256 by default)
    Document skiplist filtering (punctuation removal)

Compatible Models: ModernBERT, any encoder with (seq_len, hidden_size) output

Implements FactoryPooler protocol — zero vLLM imports.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states


class ColBERTPooler(nn.Module):
    """ColBERT pooler: projects each token to colbert_dim and L2 normalizes.

    Returns a matrix of token embeddings (seq_len, colbert_dim) per sequence,
    enabling late interaction retrieval via MaxSim scoring.
    """

    SKIPLIST = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        25, 26, 27, 28, 29, 30, 31,
        58, 59, 60, 61, 62, 63,
        90, 91, 92, 93,
    ]

    def __init__(
        self,
        hidden_size: int,
        colbert_dim: int = 128,
        normalize: bool = True,
        query_length: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.colbert_dim = colbert_dim
        self.normalize = normalize
        self.query_length = query_length
        self.linear = nn.Linear(hidden_size, colbert_dim, bias=False)

    # ── FactoryPooler protocol ───────────────────────────────────────────

    def get_tasks(self) -> set[str]:
        return {"token_embed", "embed", "plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        sequences = split_hidden_states(hidden_states, ctx.seq_lengths)
        outputs: List[torch.Tensor] = []

        for i, seq_tokens in enumerate(sequences):
            embeddings = self.linear(seq_tokens)
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

            ek = ctx.extra_kwargs[i] if i < len(ctx.extra_kwargs) else {}
            is_query = ek.get("is_query", False)
            input_ids = ek.get("input_ids", None)
            query_expansion = ek.get("query_expansion", False)

            if is_query:
                seq_len = embeddings.shape[0]
                if query_expansion:
                    if seq_len > self.query_length:
                        embeddings = embeddings[: self.query_length]
                else:
                    if seq_len < self.query_length:
                        padding = torch.zeros(
                            self.query_length - seq_len,
                            self.colbert_dim,
                            device=embeddings.device,
                            dtype=embeddings.dtype,
                        )
                        embeddings = torch.cat([embeddings, padding], dim=0)
                    elif seq_len > self.query_length:
                        embeddings = embeddings[: self.query_length]
            else:
                if input_ids is not None:
                    keep_mask = torch.ones(
                        len(input_ids), dtype=torch.bool, device=embeddings.device
                    )
                    ids_tensor = torch.tensor(input_ids, device=embeddings.device)
                    for skip_id in self.SKIPLIST:
                        keep_mask &= ids_tensor != skip_id
                    embeddings = embeddings[keep_mask]

            outputs.append(embeddings.flatten())

        return outputs
