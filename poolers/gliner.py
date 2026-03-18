# GLiNER Span Pooler — Zero-shot NER via span extraction.
#
# Architecture:
#     LSTM encoder → SpanMarker (start/end projection) → class token projection
#     Produces logits of shape (L, max_width, num_classes) per sequence
#
# Compatible Models: ModernBERT (vLLM built-in or custom), mT5 encoder
# Tested vLLM: 0.15.1
#
# This is the most complex pooler — it mirrors GLiNER's forward pass exactly:
#   1. Extract class-token (entity) embeddings from prompt region
#   2. Scatter word embeddings from subword tokens
#   3. Run LSTM over word sequence
#   4. Compute span representations via start/end projection
#   5. Score spans against entity embeddings via einsum

from __future__ import annotations

import os
from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# vLLM imports
try:
    from vllm.model_executor.pooling_metadata import PoolingMetadata, PoolingTensors
except ImportError:
    from vllm.v1.pool.metadata import PoolingMetadata
    PoolingTensors = None


# PoolerOutput type
PoolerOutput = list[torch.Tensor]


# ---------- Utility components ----------

def create_projection_layer(hidden_size: int, dropout: float, out_dim: int = None) -> nn.Sequential:
    """MLP projection: hidden → 4x hidden → out."""
    if out_dim is None:
        out_dim = hidden_size
    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim),
    )


def extract_elements(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather elements from tensor using indices. tensor: [B,L,D], indices: [B,S]."""
    B, L, D = tensor.shape
    S = indices.shape[1]
    indices = indices.clamp(min=0, max=L - 1)
    return tensor.gather(1, indices.unsqueeze(-1).expand(B, S, D))


class LstmSeq2SeqEncoder(nn.Module):
    """Bidirectional LSTM encoder with bf16→fp16 optimization for CUDA."""

    def __init__(
        self, hidden_size: int, num_layers: int = 1,
        dropout: float = 0.0, bidirectional: bool = True,
        force_fp16_inference: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size // 2,
            num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional, batch_first=True,
        )
        self.force_fp16_inference = force_fp16_inference
        self._last_dev = None
        self._last_dtype = None

    def _sync_params_to(self, device: torch.device, dtype: torch.dtype):
        if self._last_dev != device or self._last_dtype != dtype:
            self.lstm.to(device=device, dtype=dtype)
            self._last_dev, self._last_dtype = device, dtype
            self.lstm.flatten_parameters()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, hidden=None) -> torch.Tensor:
        """x: [B,L,D], mask: [B,L]"""
        if mask.sum() == 0:
            return x

        B, L, D = x.shape
        in_dtype = x.dtype
        lengths = mask.sum(dim=1).cpu()

        run_fp16 = self.force_fp16_inference and x.is_cuda and in_dtype == torch.bfloat16
        lstm_dtype = torch.float16 if run_fp16 else in_dtype
        self._sync_params_to(x.device, lstm_dtype)

        # Fast path for B==1
        if B == 1:
            eff_L = max(0, min(int(lengths[0].item()), L))
            if eff_L == 0:
                return x
            x_eff = x[:, :eff_L, :]
            if run_fp16:
                x_eff = x_eff.to(torch.float16)
            out_eff, _ = self.lstm(x_eff, hidden)
            if run_fp16:
                out_eff = out_eff.to(in_dtype)
            if eff_L < L:
                pad = out_eff.new_zeros(1, L - eff_L, out_eff.size(-1))
                return torch.cat([out_eff, pad], dim=1)
            return out_eff

        # Batch path
        if run_fp16:
            x = x.to(torch.float16)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed, hidden)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=L)
        return out.to(in_dtype) if run_fp16 else out


class SpanMarkerV0(nn.Module):
    """Span marker using start/end projections (mmbert variant)."""

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.hidden_size = hidden_size
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)
        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        """h: [B,L,D], span_idx: [B,S,2] → [B,L,max_width,D]"""
        B, L, D = h.shape
        start_rep = self.project_start(h)
        end_rep = self.project_end(h)
        start_span = extract_elements(start_rep, span_idx[:, :, 0])
        end_span = extract_elements(end_rep, span_idx[:, :, 1])
        cat = torch.cat([start_span, end_span], dim=-1).relu()
        out = self.out_project(cat)
        return out.contiguous().view(B, L, self.max_width, D)


# ---------- Main GLiNER Pooler ----------

DEBUG_PERF = os.environ.get("GLINER_DEBUG_PERF", "0") == "1"


class GLiNERSpanPooler(nn.Module):
    """GLiNER span pooler: mirrors reference GLiNER forward pass.

    Set GLINER_DEBUG_PERF=1 to enable timing logs.

    Args:
        cfg: Config object with attributes:
            hidden_size, max_width, class_token_index, gliner_dropout,
            has_rnn, embed_ent_token
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = int(
            getattr(cfg, "gliner_hidden_size", None) or getattr(cfg, "hidden_size", 768)
        )
        self.max_width = int(getattr(cfg, "max_width", 15))
        self.ent_token_id = int(getattr(cfg, "class_token_index", 256000))
        self.dropout = float(getattr(cfg, "gliner_dropout", 0.3))
        self.has_rnn = bool(getattr(cfg, "has_rnn", True))
        self.mirror_gliner_forward = True
        self.embed_ent_token = bool(getattr(cfg, "embed_ent_token", True))

        self.rnn = LstmSeq2SeqEncoder(self.hidden_size, 1, 0.0, True) if self.has_rnn else None
        self.span_rep = SpanMarkerV0(self.hidden_size, self.max_width, self.dropout)
        self.prompt_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        self.debug_perf = DEBUG_PERF
        self.eval()

    def get_supported_tasks(self):
        """Return supported tasks for vLLM 0.15.x pooling runner."""
        return {"embed", "classify"}

    def get_pooling_updates(self, task=None):
        """Request token IDs to be available in pooling metadata."""
        try:
            from vllm.model_executor.layers.pooler import PoolingParamsUpdate
            return PoolingParamsUpdate(requires_token_ids=True)
        except ImportError:
            return None

    @staticmethod
    def _fit_length(embedding: torch.Tensor, mask: torch.Tensor, target_len: int):
        B, L, D = embedding.shape
        if L == target_len:
            return embedding, mask
        if L < target_len:
            pad_len = target_len - L
            return (
                torch.cat([embedding, embedding.new_zeros(B, pad_len, D)], dim=1),
                torch.cat([mask, mask.new_zeros(B, pad_len)], dim=1),
            )
        return embedding[:, :target_len], mask[:, :target_len]

    def _extract_gliner_reps(self, tok, iid, attn_mask, true_L, wmask):
        """Mirror GLiNER's prompt/word extraction."""
        device, H = tok.device, tok.size(-1)

        # Entity class positions
        class_pos = (iid == self.ent_token_id).nonzero(as_tuple=False).flatten()
        if not self.embed_ent_token:
            class_pos = class_pos + 1
        C = int(class_pos.numel())

        if C > 0:
            pe = tok.index_select(0, class_pos).unsqueeze(0)
            pe_mask = torch.ones(1, C, dtype=torch.long, device=device)
        else:
            pe = tok.new_zeros(1, 0, H)
            pe_mask = torch.zeros(1, 0, dtype=torch.long, device=device)

        # Word embeddings via scatter
        T = int(true_L)
        we = tok.new_zeros(1, T, H)
        pos = (wmask > 0).nonzero(as_tuple=False).flatten()
        if pos.numel() > 0:
            tgt = (wmask.index_select(0, pos) - 1).long()
            we[0, tgt] = tok.index_select(0, pos)

        we_mask = torch.zeros(1, T, dtype=torch.long, device=device)
        if T > 0:
            we_mask[0, :T] = 1

        return pe, pe_mask, we, we_mask

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
            return x.to(device=device, dtype=dtype, non_blocking=True) if dtype else x.to(device=device, non_blocking=True)
        t = torch.tensor(x, device=device)
        return t.to(dtype) if dtype else t

    def _derive_metadata_from_seq(self, seq_id, pooling_metadata, device):
        """Derive input_ids/words_mask/text_lengths from seq_data (HTTP path)."""
        seq_data = getattr(pooling_metadata, "seq_data", {})
        if seq_id not in seq_data:
            return None
        sd = seq_data[seq_id]
        raw_ids = getattr(sd, "prompt_token_ids",
                    getattr(sd, "_prompt_token_ids",
                      getattr(sd, "get_prompt_token_ids", lambda: None)()))
        if raw_ids is None:
            return None

        input_ids = list(raw_ids)
        L = len(input_ids)
        iid = torch.tensor(input_ids, dtype=torch.long, device=device)
        ent_positions = (iid == self.ent_token_id).nonzero(as_tuple=False).flatten()
        text_start = (ent_positions[-1].item() + 2) if ent_positions.numel() > 0 else 1

        wmask = torch.zeros(L, dtype=torch.long, device=device)
        word_idx = 1
        for pos in range(text_start, L):
            tid = input_ids[pos]
            if tid == self.ent_token_id or tid in (0, 1, 2, 3):
                continue
            wmask[pos] = word_idx
            word_idx += 1
        return {"input_ids": iid, "words_mask": wmask, "text_lengths": max(word_idx - 1, 1)}

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

    def forward(self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata) -> PoolerOutput:
        # Guard: during vLLM warmup/dummy runs, return dummy output
        try:
            sequences = self._extract_sequences(hidden_states, pooling_metadata)
        except Exception:
            # Warmup with dummy data — return a small dummy embedding
            dummy = torch.zeros(4, device=hidden_states.device, dtype=hidden_states.dtype)
            return [EmbeddingOutput(embedding=dummy)]  # noqa: F821

        # Get pooling params
        pp_list: List[Any] = []
        if hasattr(pooling_metadata, "pooling_params") and pooling_metadata.pooling_params:
            pp_list = list(pooling_metadata.pooling_params)
        elif hasattr(pooling_metadata, "seq_groups") and pooling_metadata.seq_groups:
            for seq_ids, pp in pooling_metadata.seq_groups:
                pp_list.extend([pp] * len(seq_ids))

        if not pp_list:
            return [
                torch.zeros((1, self.max_width, 1), device=hidden_states.device, dtype=torch.float32)
                for _ in sequences
            ]

        # Pad/trim
        while len(pp_list) < len(sequences):
            pp_list.append(None)
        pp_list = pp_list[:len(sequences)]

        outputs: List[torch.Tensor] = []

        for i, tok in enumerate(sequences):
            dev, _H = tok.device, tok.shape[-1]
            add = self._get_extra_kwargs(pp_list[i])

            # Derive metadata from seq_data if not in extra_kwargs
            if add is None or "input_ids" not in add:
                seq_id = None
                if hasattr(pooling_metadata, "seq_groups") and i < len(pooling_metadata.seq_groups):
                    seq_ids_group = pooling_metadata.seq_groups[i][0]
                    if seq_ids_group:
                        seq_id = seq_ids_group[0]
                if seq_id is not None:
                    add = self._derive_metadata_from_seq(seq_id, pooling_metadata, dev)

            if add is None:
                outputs.append(torch.zeros((1, self.max_width, 1), device=dev, dtype=torch.float32))
                continue

            iid = self._to_tensor(add["input_ids"], device=dev, dtype=torch.long)
            wmask = self._to_tensor(add["words_mask"], device=dev, dtype=torch.long)
            true_L = int(add["text_lengths"])

            attn_mask = add.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = self._to_tensor(attn_mask, device=dev, dtype=torch.long)

            span_idx_add = add.get("span_idx", None)
            span_mask_add = add.get("span_mask", None)

            pe_raw, pe_mask, we_raw, we_mask = self._extract_gliner_reps(
                tok, iid, attn_mask, true_L, wmask,
            )

            we = self.rnn(we_raw, we_mask) if self.rnn is not None else we_raw

            if span_idx_add is not None and span_mask_add is not None:
                span_idx = self._to_tensor(span_idx_add, device=dev, dtype=torch.long)
                span_mask = self._to_tensor(span_mask_add, device=dev, dtype=torch.long)
                if span_idx.dim() == 2:
                    span_idx = span_idx.unsqueeze(0)
                if span_mask.dim() == 1:
                    span_mask = span_mask.unsqueeze(0)
            else:
                block_L = we.size(1)
                starts = torch.arange(block_L, device=dev).repeat_interleave(self.max_width)
                widths = torch.arange(self.max_width, device=dev).repeat(block_L)
                span_idx = torch.stack([starts, starts + widths], dim=-1).unsqueeze(0)
                span_mask = ((span_idx[0, :, 0] < true_L) & (span_idx[0, :, 1] < true_L)).unsqueeze(0)

            target_W = span_idx.size(1) // self.max_width
            we, we_mask = self._fit_length(we, we_mask, target_W)

            span_idx_masked = span_idx * span_mask.unsqueeze(-1)
            span_rep = self.span_rep(we, span_idx_masked)
            pe = self.prompt_proj(pe_raw)
            scores = torch.einsum("BLKD,BCD->BLKC", span_rep, pe).squeeze(0)
            # Flatten to 1D for vLLM embedding interface (L*K*C,)
            # Prepend shape info so client can reshape: [L, K, C, scores...]
            L, K, C = scores.shape
            shape_prefix = torch.tensor([L, K, C], device=scores.device, dtype=scores.dtype)
            flat = torch.cat([shape_prefix, scores.flatten()])
            outputs.append(flat)

        return outputs
