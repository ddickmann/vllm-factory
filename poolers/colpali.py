"""
ColPali Pooler — Multi-vector embeddings for vision-language document retrieval.

Architecture:
    Linear projection: hidden_size → colpali_dim (128) with bias
    L2 normalization per token
    Attention mask application (zeros out padding)

Compatible Models: Qwen3-VL, Qwen2-VL, any VLM with token-level output

Implements FactoryPooler protocol — zero vLLM imports.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states


class ColPaliPooler(nn.Module):
    """ColPali pooler: projects tokens to colpali_dim and L2 normalizes.

    Returns multi-vector embeddings (seq_len, colpali_dim) per sequence
    for late interaction document retrieval via MaxSim.
    """

    def __init__(
        self,
        hidden_size: int,
        colpali_dim: int = 128,
        normalize: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.colpali_dim = colpali_dim
        self.normalize = normalize
        self.linear = nn.Linear(hidden_size, colpali_dim, bias=True)

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
            attention_mask = ek.get("attention_mask", None)
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(
                        attention_mask,
                        device=embeddings.device,
                        dtype=embeddings.dtype,
                    )
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(-1)
                embeddings = embeddings * attention_mask

            outputs.append(embeddings)

        return outputs
