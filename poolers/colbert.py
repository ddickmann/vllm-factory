"""
ColBERT Pooler — Token-level multi-vector embeddings for late interaction retrieval.

Architecture:
    Linear projection: hidden_size (768) → colbert_dim (128)
    L2 normalization per token
    Query padding to fixed length (256 by default)
    Document skiplist filtering (punctuation removal)

Compatible Models: ModernBERT, any encoder with (seq_len, hidden_size) output
Tested vLLM: 0.15.1
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

PoolerOutput = list[torch.Tensor]


class ColBERTPooler(nn.Module):
    """ColBERT pooler: projects each token to colbert_dim and L2 normalizes.

    Returns a matrix of token embeddings (seq_len, colbert_dim) per sequence,
    enabling late interaction retrieval via MaxSim scoring.
    """

    # PyLate skiplist: punctuation and special characters to filter from documents
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
        quant_config: object = None,
        query_length: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.colbert_dim = colbert_dim
        self.normalize = normalize
        self.query_length = query_length

        self.linear = ReplicatedLinear(
            hidden_size, colbert_dim, bias=False, quant_config=quant_config,
        )

    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"token_embed", "embed"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=False)

    def _extract_sequences(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> List[torch.Tensor]:
        """Split concatenated hidden states into per-sequence tensors."""
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[0]
        if hidden_states.dim() == 3:
            if hidden_states.shape[0] == 1:
                hidden_states = hidden_states.squeeze(0)
            else:
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        prompt_lens = pooling_metadata.prompt_lens.to(hidden_states.device).tolist()
        sequences, offset = [], 0
        for seq_len in prompt_lens:
            sequences.append(hidden_states[offset:offset + seq_len])
            offset += seq_len
        return sequences

    @staticmethod
    def _get_extra_kwargs(pooling_params) -> Optional[dict]:
        """Extract extra_kwargs from pooling params (offline or HTTP API)."""
        for attr in ("extra_kwargs", "additional_data", "additional_metadata"):
            md = getattr(pooling_params, attr, None)
            if md is not None and isinstance(md, dict):
                return md
        return None

    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        sequences = self._extract_sequences(hidden_states, pooling_metadata)

        # Get pooling params
        pooling_params_list = []
        if hasattr(pooling_metadata, "pooling_params") and pooling_metadata.pooling_params:
            pooling_params_list = list(pooling_metadata.pooling_params)
        elif hasattr(pooling_metadata, "seq_groups") and pooling_metadata.seq_groups:
            for seq_ids, pp in pooling_metadata.seq_groups:
                pooling_params_list.extend([pp] * len(seq_ids))

        outputs: List[torch.Tensor] = []

        for i, seq_tokens in enumerate(sequences):
            embeddings, _ = self.linear(seq_tokens)
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

            # Extract metadata
            is_query, input_ids, query_expansion = False, None, False
            if i < len(pooling_params_list) and pooling_params_list[i] is not None:
                md = self._get_extra_kwargs(pooling_params_list[i])
                if md:
                    is_query = md.get("is_query", False)
                    input_ids = md.get("input_ids", None)
                    query_expansion = md.get("query_expansion", False)

            if is_query:
                seq_len = embeddings.shape[0]
                if query_expansion:
                    if seq_len > self.query_length:
                        embeddings = embeddings[:self.query_length]
                else:
                    if seq_len < self.query_length:
                        padding = torch.zeros(
                            self.query_length - seq_len, self.colbert_dim,
                            device=embeddings.device, dtype=embeddings.dtype,
                        )
                        embeddings = torch.cat([embeddings, padding], dim=0)
                    elif seq_len > self.query_length:
                        embeddings = embeddings[:self.query_length]
            else:
                if input_ids is not None:
                    keep_mask = torch.ones(len(input_ids), dtype=torch.bool, device=embeddings.device)
                    ids_tensor = torch.tensor(input_ids, device=embeddings.device)
                    for skip_id in self.SKIPLIST:
                        keep_mask &= (ids_tensor != skip_id)
                    embeddings = embeddings[keep_mask]

            outputs.append(embeddings.flatten())

        return outputs
