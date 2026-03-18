"""
GLiNER-Linker Token Pooler — token-level entity linking scoring.

Mirrors GLiNER BiEncoderTokenModel forward AFTER the text encoder (same as
``gliner.modeling.base.BiEncoderTokenModel.forward``): word extraction then
scorer. The checkpoint includes an unused ``rnn`` submodule for this
architecture; GLiNER inference does not apply it before the scorer.
1. Split concatenated hidden states into per-sequence tensors
2. Extract word-level embeddings using words_mask
3. Encode labels (or use precomputed label embeddings)
4. Run Scorer → (W, C, 3) logits (start, end, inside scores)
5. Flatten + prepend shape info for vLLM embedding interface

Input metadata comes via PoolingParams.extra_kwargs:
    - words_mask: [L] int, 1-indexed word assignments
    - text_lengths: int, number of words
    - attention_mask: [L] int, full-prompt mask (same length as `prompt_token_ids`); forwarded
      into the text encoder forward by vLLM runner patch + `GLiNERLinkerModel.forward`
    - label_texts: list[str], entity labels to encode (if not using precomputed)
    - labels_embeds: (C, D) tensor, precomputed label embeddings (optional)
"""

from __future__ import annotations

from typing import Any, List, Optional, Set

import torch
import torch.nn as nn

try:
    from vllm.model_executor.layers.pooler import PoolingTensors
except ImportError:
    PoolingTensors = None

try:
    from vllm.v1.pool.metadata import PoolingMetadata
except ImportError:
    try:
        from vllm.model_executor.layers.pooler import PoolingMetadata
    except ImportError:
        PoolingMetadata = None


PoolerOutput = list[torch.Tensor]


class GLiNERLinkerPooler(nn.Module):
    """Token-level pooler for GLiNER-Linker bi-encoder.

    Receives text encoder hidden states, extracts word embeddings, runs the
    scorer against label embeddings, returns flattened logits (no LSTM on this path).
    """

    def __init__(self, model):
        """Initialize with references to individual model components.

        Stores component references (not the parent model) to avoid
        circular references that cause RecursionError on model.eval().

        Args:
            model: GLiNERLinkerModel instance with labels_encoder, scorer heads
        """
        super().__init__()
        # Use object.__setattr__ to avoid nn.Module submodule registration
        # which would create circular references (model → pooler → model's children)
        object.__setattr__(self, '_labels_encoder', model.labels_encoder)
        object.__setattr__(self, '_scorer_proj_token', model.scorer_proj_token)
        object.__setattr__(self, '_scorer_proj_label', model.scorer_proj_label)
        object.__setattr__(self, '_scorer_out_mlp', model.scorer_out_mlp)
        object.__setattr__(self, '_model_config', model.vllm_config.model_config)
        # Cached tokenizer (loaded lazily on first use)
        object.__setattr__(self, '_tokenizer', None)

    def get_supported_tasks(self) -> Set[str]:
        return {"embed"}

    def get_pooling_updates(self, task=None):
        """Request token IDs in pooling metadata."""
        from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
        return PoolingParamsUpdate(requires_token_ids=True)

    @staticmethod
    def _get_extra_kwargs(pp) -> Optional[dict]:
        for attr in ("extra_kwargs", "additional_data", "additional_metadata"):
            md = getattr(pp, attr, None)
            if md is not None and isinstance(md, dict):
                return md
        return None

    @staticmethod
    def _to_tensor(x, device, dtype=None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype) if dtype else x.to(device=device)
        t = torch.tensor(x, device=device)
        return t.to(dtype) if dtype else t

    def _extract_sequences(self, hidden_states, pooling_metadata):
        """Split concatenated hidden states into per-sequence tensors."""
        if PoolingTensors is not None:
            prompt_lens = PoolingTensors.from_pooling_metadata(
                pooling_metadata, hidden_states.device
            ).prompt_lens
        else:
            prompt_lens = pooling_metadata.prompt_lens.to(hidden_states.device)

        sequences, offset = [], 0
        for L in prompt_lens:
            sequences.append(hidden_states[offset:offset + L])
            offset += L
        return sequences

    def _extract_word_embeddings(self, tok_embs, wmask, text_length, device, H):
        """Extract word-level embeddings from subtoken embeddings.

        Args:
            tok_embs: (L, H) subtoken embeddings
            wmask: (L,) int tensor, 1-indexed word positions
            text_length: int, number of words
            device: target device
            H: hidden dimension

        Returns:
            word_embs: (1, W, H) word-level embeddings
            word_mask: (1, W) binary mask
        """
        W = int(text_length)
        we = tok_embs.new_zeros(1, W, H)
        pos = (wmask > 0).nonzero(as_tuple=False).flatten()
        if pos.numel() > 0:
            tgt = (wmask[pos] - 1).long()
            we[0, tgt] = tok_embs[pos]
        we_mask = torch.ones(1, W, dtype=torch.long, device=device)
        return we, we_mask

    @torch.no_grad()
    def _encode_labels(self, label_texts: List[str], device) -> torch.Tensor:
        """Encode label texts using the labels encoder + mean pooling.

        Args:
            label_texts: list of entity label strings
            device: target device

        Returns:
            (C, H) tensor of label embeddings
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            object.__setattr__(
                self, '_tokenizer',
                AutoTokenizer.from_pretrained(self._model_config.model, use_fast=True),
            )

        all_embs = []
        for label in label_texts:
            enc = self._tokenizer(label, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            output = self._labels_encoder(input_ids=input_ids)
            hs = output.last_hidden_state  # (1, L, H)

            # Mean pool
            mask_expanded = attention_mask.unsqueeze(-1).expand(hs.size()).float()
            mean = (hs * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
            all_embs.append(mean.squeeze(0))  # (H,)

        return torch.stack(all_embs, dim=0)  # (C, H)

    def _run_scorer(self, word_embs, label_embs):
        """Run the scorer head.

        Args:
            word_embs: (1, W, H) word-level embeddings
            label_embs: (1, C, H) label embeddings

        Returns:
            scores: (1, W, C, 3) logits
        """
        B, W, H = word_embs.shape
        C = label_embs.shape[1]

        # Project and split into two components for bilinear interaction
        token_rep = self._scorer_proj_token(word_embs)  # (1, W, 2H)
        token_rep = token_rep.view(B, W, 1, 2, H)

        label_rep = self._scorer_proj_label(label_embs)  # (1, C, 2H)
        label_rep = label_rep.view(B, 1, C, 2, H)

        # Expand for pairwise computation
        token_rep = token_rep.expand(-1, -1, C, -1, -1).permute(3, 0, 1, 2, 4)
        label_rep = label_rep.expand(-1, W, -1, -1, -1).permute(3, 0, 1, 2, 4)

        # Concatenate: [first_token_proj, first_label_proj, element_wise_product]
        cat = torch.cat([token_rep[0], label_rep[0], token_rep[1] * label_rep[1]], dim=-1)

        # Final scores through MLP
        scores = self._scorer_out_mlp(cat)  # (1, W, C, 3)
        return scores

    def forward(self, hidden_states: torch.Tensor, pooling_metadata) -> PoolerOutput:
        """Full token-level pooler forward pass.

        Args:
            hidden_states: concatenated text encoder hidden states (total_tokens, H)
            pooling_metadata: contains prompt_lens and pooling_params

        Returns:
            List of flattened score tensors, one per sequence
        """
        # Guard: during warmup/dummy runs
        try:
            sequences = self._extract_sequences(hidden_states, pooling_metadata)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Pooler warmup fallback: %s", e)
            dummy = torch.zeros(4, device=hidden_states.device, dtype=hidden_states.dtype)
            return [dummy]

        # Get pooling params
        pp_list: List[Any] = []
        if hasattr(pooling_metadata, "pooling_params") and pooling_metadata.pooling_params:
            pp_list = list(pooling_metadata.pooling_params)
        elif hasattr(pooling_metadata, "seq_groups") and pooling_metadata.seq_groups:
            for seq_ids, pp in pooling_metadata.seq_groups:
                pp_list.extend([pp] * len(seq_ids))

        if not pp_list:
            return [
                torch.zeros(4, device=hidden_states.device, dtype=torch.float32)
                for _ in sequences
            ]

        while len(pp_list) < len(sequences):
            pp_list.append(None)
        pp_list = pp_list[:len(sequences)]

        outputs: List[torch.Tensor] = []
        dev = hidden_states.device
        H = hidden_states.shape[-1]

        for i, tok in enumerate(sequences):
            add = self._get_extra_kwargs(pp_list[i])

            if add is None or "words_mask" not in add:
                outputs.append(torch.zeros(4, device=dev, dtype=torch.float32))
                continue

            wmask = self._to_tensor(add["words_mask"], device=dev, dtype=torch.long)
            text_length = int(add["text_lengths"])

            # 1. Extract word-level embeddings (matches GLiNER extract_word_embeddings path)
            we, _ = self._extract_word_embeddings(tok, wmask, text_length, dev, H)

            # 2. Get label embeddings
            if "labels_embeds" in add:
                le = self._to_tensor(add["labels_embeds"], device=dev, dtype=we.dtype)
                if le.dim() == 2:
                    le = le.unsqueeze(0)  # (1, C, H)
            elif "label_texts" in add:
                le_flat = self._encode_labels(add["label_texts"], dev)
                le = le_flat.unsqueeze(0).to(we.dtype)  # (1, C, H)
            else:
                outputs.append(torch.zeros(4, device=dev, dtype=torch.float32))
                continue

            # 3. Run scorer
            scores = self._run_scorer(we, le)  # (1, W, C, 3)
            scores = scores.squeeze(0)  # (W, C, 3)

            # 4. Flatten + prepend shape info for client decoding
            W, C, S = scores.shape
            shape_prefix = torch.tensor([W, C, S], device=dev, dtype=scores.dtype)
            flat = torch.cat([shape_prefix, scores.flatten()])
            outputs.append(flat)

        return outputs
