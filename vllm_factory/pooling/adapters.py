"""vLLM pooler adapters — translate PoolingMetadata into PoolerContext.

Each adapter wraps a vLLM nn.Module pooler's ``forward()`` call,
translating the raw ``(hidden_states, pooling_metadata)`` pair into
the stable ``PoolerContext`` before dispatching to the pooler kernel.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from vllm_factory.pooling.context import PoolerContext, SequenceContext


def extract_sequences(
    hidden_states: torch.Tensor,
    pooling_metadata: Any,
) -> list[torch.Tensor]:
    """Split concatenated hidden states into per-sequence tensors."""
    if isinstance(hidden_states, list):
        hidden_states = hidden_states[0]
    if hidden_states.dim() == 3:
        if hidden_states.shape[0] == 1:
            hidden_states = hidden_states.squeeze(0)
        else:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    prompt_lens: list[int] = []
    if hasattr(pooling_metadata, "prompt_lens"):
        prompt_lens = pooling_metadata.prompt_lens.to(hidden_states.device).tolist()
    else:
        try:
            from vllm.model_executor.pooling_metadata import PoolingTensors
            pt = PoolingTensors.from_pooling_metadata(
                pooling_metadata, hidden_states.device
            )
            prompt_lens = pt.prompt_lens.tolist()
        except Exception:
            prompt_lens = [hidden_states.shape[0]]

    sequences: list[torch.Tensor] = []
    offset = 0
    for seq_len in prompt_lens:
        sequences.append(hidden_states[offset:offset + seq_len])
        offset += seq_len
    return sequences


def get_extra_kwargs(pooling_params: Any) -> Optional[dict]:
    """Extract extra_kwargs dict from a single PoolingParams instance."""
    if pooling_params is None:
        return None
    for attr in ("extra_kwargs", "additional_data", "additional_metadata"):
        md = getattr(pooling_params, attr, None)
        if md is not None and isinstance(md, dict):
            return md
    return None


def get_pooling_params_list(pooling_metadata: Any) -> list[Any]:
    """Get the list of per-sequence PoolingParams from metadata."""
    pp_list: list[Any] = []
    if hasattr(pooling_metadata, "pooling_params") and pooling_metadata.pooling_params:
        pp_list = list(pooling_metadata.pooling_params)
    elif hasattr(pooling_metadata, "seq_groups") and pooling_metadata.seq_groups:
        for seq_ids, pp in pooling_metadata.seq_groups:
            pp_list.extend([pp] * len(seq_ids))
    return pp_list


def build_pooler_context(
    hidden_states: torch.Tensor,
    pooling_metadata: Any,
) -> PoolerContext:
    """Build a PoolerContext from vLLM's raw (hidden_states, pooling_metadata)."""
    seqs = extract_sequences(hidden_states, pooling_metadata)
    pp_list = get_pooling_params_list(pooling_metadata)

    while len(pp_list) < len(seqs):
        pp_list.append(None)
    pp_list = pp_list[:len(seqs)]

    contexts = []
    for i, seq_hs in enumerate(seqs):
        ek = get_extra_kwargs(pp_list[i]) or {}
        contexts.append(SequenceContext(
            hidden_states=seq_hs,
            extra_kwargs=ek,
            seq_len=seq_hs.shape[0],
            token_ids=ek.get("input_ids"),
        ))

    return PoolerContext(
        sequences=contexts,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
