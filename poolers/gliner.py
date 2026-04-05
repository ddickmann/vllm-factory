# GLiNER Span Pooler — Zero-shot NER via span extraction.
#
# Architecture:
#     LSTM encoder → SpanMarker (start/end projection) → class token projection
#     Produces logits of shape (L, max_width, num_classes) per sequence
#
# Compatible Models: ModernBERT (vLLM built-in or custom), mT5 encoder
#
# Implements FactoryPooler protocol — zero vLLM imports.

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states

# ---------- Utility components ----------


def create_projection_layer(
    hidden_size: int, dropout: float, out_dim: int = None
) -> nn.Sequential:
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
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        force_fp16_inference: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
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

        run_fp16 = (
            self.force_fp16_inference and x.is_cuda and in_dtype == torch.bfloat16
        )
        lstm_dtype = torch.float16 if run_fp16 else in_dtype
        self._sync_params_to(x.device, lstm_dtype)

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

        if run_fp16:
            x = x.to(torch.float16)
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
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
        self.out_project = create_projection_layer(
            hidden_size * 2, dropout, hidden_size
        )

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
            getattr(cfg, "gliner_hidden_size", None)
            or getattr(cfg, "hidden_size", 768)
        )
        self.max_width = int(getattr(cfg, "max_width", 15))
        self.ent_token_id = int(getattr(cfg, "class_token_index", 256000))
        self.dropout = float(getattr(cfg, "gliner_dropout", 0.3))
        self.has_rnn = bool(getattr(cfg, "has_rnn", True))
        self.mirror_gliner_forward = True
        self.embed_ent_token = bool(getattr(cfg, "embed_ent_token", True))

        self.rnn = (
            LstmSeq2SeqEncoder(self.hidden_size, 1, 0.0, True)
            if self.has_rnn
            else None
        )
        self.span_rep = SpanMarkerV0(
            self.hidden_size, self.max_width, self.dropout
        )
        self.prompt_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        self.debug_perf = DEBUG_PERF
        self.eval()

    # ── FactoryPooler protocol ───────────────────────────────────────────

    def get_tasks(self) -> set[str]:
        return {"embed", "classify", "plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        try:
            sequences = split_hidden_states(hidden_states, ctx.seq_lengths)
        except Exception:
            dummy = torch.zeros(
                4, device=hidden_states.device, dtype=hidden_states.dtype
            )
            return [dummy]

        if not ctx.extra_kwargs:
            return [
                torch.zeros(
                    (1, self.max_width, 1),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                for _ in sequences
            ]

        outputs: List[torch.Tensor] = []

        for i, tok in enumerate(sequences):
            dev = tok.device
            add = ctx.extra_kwargs[i] if i < len(ctx.extra_kwargs) else {}
            prompt_ids = ctx.prompt_token_ids[i] if i < len(ctx.prompt_token_ids) else None

            if not add:
                outputs.append(
                    torch.zeros(
                        (1, self.max_width, 1), device=dev, dtype=torch.float32
                    )
                )
                continue

            input_ids_value = add.get("input_ids", prompt_ids)
            if input_ids_value is None:
                outputs.append(
                    torch.zeros(
                        (1, self.max_width, 1), device=dev, dtype=torch.float32
                    )
                )
                continue

            iid = self._to_tensor(input_ids_value, device=dev, dtype=torch.long)
            wmask = self._to_tensor(add["words_mask"], device=dev, dtype=torch.long)
            true_L = int(add["text_lengths"])

            attn_mask = add.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = self._to_tensor(attn_mask, device=dev, dtype=torch.long)

            span_idx_add = add.get("span_idx", None)
            span_mask_add = add.get("span_mask", None)

            pe_raw, pe_mask, we_raw, we_mask = self._extract_gliner_reps(
                tok, iid, attn_mask, true_L, wmask
            )

            we = self.rnn(we_raw, we_mask) if self.rnn is not None else we_raw

            if span_idx_add is not None and span_mask_add is not None:
                span_idx = self._to_tensor(
                    span_idx_add, device=dev, dtype=torch.long
                )
                span_mask = self._to_tensor(
                    span_mask_add, device=dev, dtype=torch.long
                )
                if span_idx.dim() == 2:
                    span_idx = span_idx.unsqueeze(0)
                if span_mask.dim() == 1:
                    span_mask = span_mask.unsqueeze(0)
            else:
                block_L = we.size(1)
                starts = torch.arange(block_L, device=dev).repeat_interleave(
                    self.max_width
                )
                widths = torch.arange(self.max_width, device=dev).repeat(block_L)
                span_idx = torch.stack(
                    [starts, starts + widths], dim=-1
                ).unsqueeze(0)
                span_mask = (
                    (span_idx[0, :, 0] < true_L) & (span_idx[0, :, 1] < true_L)
                ).unsqueeze(0)

            target_W = span_idx.size(1) // self.max_width
            we, we_mask = self._fit_length(we, we_mask, target_W)

            span_idx_masked = span_idx * span_mask.unsqueeze(-1)
            span_rep = self.span_rep(we, span_idx_masked)
            pe = self.prompt_proj(pe_raw)
            scores = torch.einsum("BLKD,BCD->BLKC", span_rep, pe).squeeze(0)
            L, K, C = scores.shape
            shape_prefix = torch.tensor(
                [L, K, C], device=scores.device, dtype=scores.dtype
            )
            flat = torch.cat([shape_prefix, scores.flatten()])
            outputs.append(flat)

        return outputs

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _fit_length(embedding: torch.Tensor, mask: torch.Tensor, target_len: int):
        B, L, D = embedding.shape
        if L == target_len:
            return embedding, mask
        if L < target_len:
            pad_len = target_len - L
            return (
                torch.cat(
                    [embedding, embedding.new_zeros(B, pad_len, D)], dim=1
                ),
                torch.cat([mask, mask.new_zeros(B, pad_len)], dim=1),
            )
        return embedding[:, :target_len], mask[:, :target_len]

    def _extract_gliner_reps(self, tok, iid, attn_mask, true_L, wmask):
        """Mirror GLiNER's prompt/word extraction."""
        device, H = tok.device, tok.size(-1)

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

        T = int(true_L)
        we = tok.new_zeros(1, T, H)
        pos = (wmask > 0).nonzero(as_tuple=False).flatten()
        if pos.numel() > 0:
            tgt = (wmask.index_select(0, pos) - 1).long()
            # GLiNER token models use subtoken_pooling="first". Since
            # `words_mask` repeats the same word id across all subtokens, a
            # direct indexed write would incorrectly keep the last subtoken.
            keep = torch.ones_like(tgt, dtype=torch.bool)
            if tgt.numel() > 1:
                keep[1:] = tgt[1:] != tgt[:-1]
            we[0, tgt[keep]] = tok.index_select(0, pos[keep])

        we_mask = torch.zeros(1, T, dtype=torch.long, device=device)
        if T > 0:
            we_mask[0, :T] = 1

        return pe, pe_mask, we, we_mask

    @staticmethod
    def _to_tensor(x, device, dtype=None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return (
                x.to(device=device, dtype=dtype, non_blocking=True)
                if dtype
                else x.to(device=device, non_blocking=True)
            )
        t = torch.tensor(x, device=device)
        return t.to(dtype) if dtype else t
