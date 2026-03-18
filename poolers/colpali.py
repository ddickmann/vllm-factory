"""
ColPali Pooler — Multi-vector embeddings for vision-language document retrieval.

Architecture:
    Linear projection: hidden_size → colpali_dim (128) with bias
    L2 normalization per token
    Attention mask application (zeros out padding)

Compatible Models: Qwen3-VL, Qwen2-VL, any VLM with token-level output
Tested vLLM: 0.15.1
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.linear import ReplicatedLinear

try:
    from vllm.model_executor.pooling_metadata import PoolingMetadata, PoolingTensors
except ImportError:
    from vllm.v1.pool.metadata import PoolingMetadata
    PoolingTensors = None

PoolerOutput = list[torch.Tensor]


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
        quant_config: object = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.colpali_dim = colpali_dim
        self.normalize = normalize

        self.linear = ReplicatedLinear(
            hidden_size, colpali_dim, bias=True, quant_config=quant_config,
        )

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

        if hasattr(pooling_metadata, "prompt_lens"):
            lens = pooling_metadata.prompt_lens
            prompt_lens = lens.tolist() if hasattr(lens, 'tolist') else list(lens)
        elif PoolingTensors is not None:
            tensors = PoolingTensors.from_pooling_metadata(
                pooling_metadata, hidden_states.device
            )
            prompt_lens = tensors.prompt_lens.tolist()
        else:
            raise RuntimeError("Cannot extract prompt_lens from pooling_metadata")

        sequences, offset = [], 0
        for seq_len in prompt_lens:
            sequences.append(hidden_states[offset:offset + seq_len])
            offset += seq_len
        return sequences

    @staticmethod
    def _get_extra_kwargs(pooling_params) -> Optional[dict]:
        for attr in ("extra_kwargs", "additional_data", "additional_metadata"):
            md = getattr(pooling_params, attr, None)
            if md is not None and isinstance(md, dict):
                return md
        return None

    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        sequences = self._extract_sequences(hidden_states, pooling_metadata)

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

            # Apply attention mask if available
            if i < len(pooling_params_list) and pooling_params_list[i] is not None:
                md = self._get_extra_kwargs(pooling_params_list[i])
                if md:
                    attention_mask = md.get("attention_mask", None)
                    if attention_mask is not None:
                        if not isinstance(attention_mask, torch.Tensor):
                            attention_mask = torch.tensor(
                                attention_mask, device=embeddings.device, dtype=embeddings.dtype,
                            )
                        if attention_mask.dim() == 1:
                            attention_mask = attention_mask.unsqueeze(-1)
                        embeddings = embeddings * attention_mask

            outputs.append(embeddings)

        return outputs
