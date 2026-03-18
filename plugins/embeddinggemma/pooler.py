"""EmbeddingGemma Pooler — MEAN pool + Dense1 + Dense2 + L2 normalize.

Mirrors the SentenceTransformers pipeline exactly.
"""

from __future__ import annotations

from typing import List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.v1.pool.metadata import PoolingMetadata


class EmbeddingGemmaPooler(nn.Module):
    """Custom pooler for EmbeddingGemma.

    Pipeline: MEAN pooling → Dense1 (768→3072) → Dense2 (3072→768) → L2 normalize
    """

    def __init__(self, hidden_size: int = 768, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        # Dense layers (weights loaded from HF Hub safetensors in model.load_weights)
        self.dense1 = nn.Linear(hidden_size, 3072, bias=False, dtype=dtype)
        self.dense2 = nn.Linear(3072, hidden_size, bias=False, dtype=dtype)

    def get_supported_tasks(self) -> Set[str]:
        return {"embed"}

    def get_pooling_updates(self, task=None):
        try:
            from vllm.model_executor.layers.pooler import PoolingParamsUpdate
            return PoolingParamsUpdate()
        except ImportError:
            return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> List[torch.Tensor]:
        """Apply MEAN pooling + Dense projections + L2 normalize.

        Args:
            hidden_states: (total_tokens, hidden_size) — all sequences concatenated
            pooling_metadata: Contains prompt_lens for splitting sequences

        Returns:
            List of 1D embedding tensors, one per sequence
        """
        prompt_lens = pooling_metadata.prompt_lens

        # Split by sequence and MEAN pool each
        outputs = []
        offset = 0
        for seq_len in prompt_lens:
            seq_hidden = hidden_states[offset:offset + seq_len]  # (seq_len, hidden)

            # MEAN pool (matching SentenceTransformers 1_Pooling behavior)
            pooled = seq_hidden.mean(dim=0, keepdim=True)  # (1, hidden)

            # Dense projections (matching 2_Dense and 3_Dense)
            projected = self.dense1(pooled)
            projected = self.dense2(projected)

            # L2 normalize (matching 4_Normalize)
            normalized = F.normalize(projected, p=2, dim=-1)

            outputs.append(normalized.squeeze(0))  # (hidden,)
            offset += seq_len

        return outputs
