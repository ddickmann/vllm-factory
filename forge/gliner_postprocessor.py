"""
GLiNER Postprocessor — shared decoding for GLiNER span extraction.

Mirrors superpod's gliner_postprocessor.py exactly.
Used by both mmbert_gliner and mt5_gliner plugins.

Key features:
1. Vectorized threshold filtering via torch.where
2. Greedy NMS with nested/flat modes
3. Character-offset entity extraction
"""

from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch

# --- Overlap predicates -------------------------------------------------------

def _has_overlapping(span1: Tuple[int,int,str], span2: Tuple[int,int,str], multi_label: bool) -> bool:
    s1, e1, l1 = span1; s2, e2, l2 = span2  # noqa: E702
    if multi_label and l1 == l2:
        return False
    return not (e1 < s2 or s1 > e2)


def _has_overlapping_nested(span1: Tuple[int,int,str], span2: Tuple[int,int,str], multi_label: bool) -> bool:
    s1, e1, l1 = span1; s2, e2, l2 = span2  # noqa: E702
    if multi_label and l1 == l2:
        return False
    if s1 == s2 and e1 == e2:
        return True
    if (s1 >= s2 and e1 <= e2) or (s2 >= s1 and e2 <= e1):
        return False
    return not (e1 < s2 or s1 > e2)


# --- Fast, spec-accurate decoder ---------------------------------------------

class GLiNERDecoder:
    """
    Drop-in, faster mirror of GLiNER's SpanDecoder.decode:
      - logits shape: (B, L, K, C)  (raw logits)
      - id_to_classes: dict[int->str] or list[dict] (indexing starts at 1)
      - tokens: list[list[str]]  (for sequence-length bounds)
      - returns: list[list[(start, end, ent_type, gen_ent_type, score)]]
    """
    def __init__(self): pass

    @staticmethod
    def _build_span_label_maps(
        sel_idx: Optional[torch.Tensor],
        gen_labels: Optional[List[str]],
        num_gen_sequences: int,
    ) -> List[Dict[int, List[str]]]:
        """Build per-batch mapping: flat_idx -> [generated labels]."""
        if sel_idx is None or gen_labels is None:
            return []
        B = sel_idx.size(0)
        cursor = 0
        out: List[Dict[int, List[str]]] = [{} for _ in range(B)]
        for b in range(B):
            valid = (sel_idx[b] != -1)
            n = int(valid.sum().item())
            if n == 0:
                continue
            flat_indices = sel_idx[b, valid].tolist()
            start = cursor * num_gen_sequences
            span_lbls = gen_labels[start : start + n * num_gen_sequences]
            grouped = [span_lbls[i * num_gen_sequences:(i + 1) * num_gen_sequences] for i in range(n)]
            out[b] = dict(zip(flat_indices, grouped))
            cursor += n
        return out

    @staticmethod
    def _greedy_nms(
        spans: List[Tuple[int,int,str,Optional[List[str]],float]],
        flat_ner: bool,
        multi_label: bool
    ) -> List[Tuple[int,int,str,Optional[List[str]],float]]:
        """Greedy NMS: sort by score desc, remove overlapping spans."""
        has_ov = partial(_has_overlapping, multi_label=multi_label) if flat_ner \
                 else partial(_has_overlapping_nested, multi_label=multi_label)
        spans_sorted = sorted(spans, key=lambda x: -x[-1])
        keep = []
        for cand in spans_sorted:
            s,e,lab,genlab,score = cand
            clash = False
            for k in keep:
                if has_ov((s,e,lab), (k[0],k[1],k[2])):
                    clash = True
                    break
            if not clash:
                keep.append(cand)
        return sorted(keep, key=lambda x: x[0])

    @torch.no_grad()
    def decode(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int,str], List[Dict[int,str]]],
        logits: torch.Tensor,              # (B,L,K,C) raw logits
        flat_ner: bool = False,
        threshold: float = 0.5,
        multi_label: bool = False,
        sel_idx: Optional[torch.Tensor] = None,
        gen_labels: Optional[List[str]] = None,
        num_gen_sequences: int = 1,
    ) -> List[List[Tuple[int,int,str,Optional[List[str]],float]]]:

        B, L, K, C = logits.shape
        device = logits.device
        probs = logits.sigmoid()

        b_idx, s_idx, k_idx, c_idx = torch.where(probs > threshold)
        if b_idx.numel() == 0:
            return [[] for _ in range(B)]

        scores = probs[b_idx, s_idx, k_idx, c_idx]
        end_exclusive = s_idx + k_idx + 1
        seq_len = torch.as_tensor([len(t) for t in tokens], device=device)
        valid = end_exclusive <= seq_len[b_idx]

        if not valid.any():
            return [[] for _ in range(B)]

        b_idx, s_idx, k_idx, c_idx, scores, end_exclusive = (
            b_idx[valid], s_idx[valid], k_idx[valid], c_idx[valid], scores[valid], end_exclusive[valid]
        )
        end_inclusive = end_exclusive - 1
        flat_idx = s_idx * K + k_idx

        # Bulk GPU -> CPU transfer
        b_cpu = b_idx.tolist()
        s_cpu = s_idx.tolist()
        e_cpu = end_inclusive.tolist()
        c_cpu = c_idx.tolist()
        sc_cpu = scores.tolist()
        f_cpu = flat_idx.tolist()

        # Pre-bucket candidates by batch index
        batched_candidates: List[List[Tuple[int,int,int,float,int]]] = [[] for _ in range(B)]
        for s, e, c, sc, f, b in zip(s_cpu, e_cpu, c_cpu, sc_cpu, f_cpu, b_cpu):
            batched_candidates[b].append((s, e, c, sc, f))

        span_label_maps = self._build_span_label_maps(sel_idx, gen_labels, num_gen_sequences)

        out: List[List[Tuple[int,int,str,Optional[List[str]],float]]] = [[] for _ in range(B)]
        for i in range(B):
            candidates_i = batched_candidates[i]
            if not candidates_i:
                continue

            id2c = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes
            spans_i = []

            if span_label_maps:
                mp = span_label_maps[i]
                for s, e, c, sc, f in candidates_i:
                    ent = id2c[c + 1]
                    gen = mp.get(f) if mp else None
                    spans_i.append((s, e, ent, gen, float(sc)))
            else:
                for s, e, c, sc, f in candidates_i:
                    ent = id2c[c + 1]
                    spans_i.append((s, e, ent, None, float(sc)))

            out[i] = self._greedy_nms(spans_i, flat_ner=flat_ner, multi_label=multi_label)

        return out


def get_final_entities(
    decoded_outputs: list,
    word_positions: list,
    original_texts: list
) -> list:
    """
    Convert token-based entity indices into character-based indices.
    Mirrors GLiNER.run() method's final post-processing step.
    """
    all_start_token_idx_to_text_idx = []
    all_end_token_idx_to_text_idx = []

    for batch_item_positions in word_positions:
        all_start_token_idx_to_text_idx.append(
            [pos[0] for pos in batch_item_positions]
        )
        all_end_token_idx_to_text_idx.append(
            [pos[1] for pos in batch_item_positions]
        )

    all_entities = []
    for i, output in enumerate(decoded_outputs):
        text = original_texts[i]
        start_token_idx_to_text_idx = all_start_token_idx_to_text_idx[i]
        end_token_idx_to_text_idx = all_end_token_idx_to_text_idx[i]

        entities = []
        for start_token_idx, end_token_idx, ent_type, gen_ent_type, ent_score in output:
            start_text_idx = start_token_idx_to_text_idx[start_token_idx]
            end_text_idx = end_token_idx_to_text_idx[end_token_idx]
            entity_text = text[start_text_idx:end_text_idx]

            ent_details = {
                "start": start_text_idx,
                "end": end_text_idx,
                "text": entity_text,
                "label": ent_type,
                "score": ent_score,
            }
            if gen_ent_type is not None:
                ent_details['generated labels'] = gen_ent_type
            entities.append(ent_details)

        all_entities.append(entities)

    return all_entities
