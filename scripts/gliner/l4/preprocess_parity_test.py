#!/usr/bin/env python3
"""
Tensor-level parity: GLiNER TokenDataCollator (uni-encoder) vs GLiNERRerankProcessor._tokenize.

Skips vLLM (monkeypatches _ensure_llm). CPU-safe.

Checks:
  1) Single-example batch (legacy): full-tensor equality.
  2) Multi-example batch: collator right-pads to max length; each row must match the
     per-text ``_tokenize`` tensors truncated to ``attention_mask.sum()`` (same pattern
     as ``batch_predict_entities``, which tokenizes each string separately).

Usage:
    cd vllm-factory && PYTHONPATH=. python scripts/gliner/l4/preprocess_parity_test.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import torch

from l4_parity_fixtures import HF_MODEL_ID, MULTI_TEXTS, TEST_LABELS, TEST_TEXT

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _valid_token_len(attention_mask: torch.Tensor) -> int:
    return int(attention_mask.sum().item())


def _word_count_row(text_lengths: torch.Tensor, row: int) -> int:
    tl = text_lengths
    if tl.dim() == 2:
        return int(tl[row, 0].item())
    return int(tl[row].item())


def _words_for_text(data_processor, text: str) -> List[str]:
    return [t for t, _s, _e in data_processor.words_splitter(text)]


def reference_batch_from_rows(
    rows: List[dict],
    labels: List[str],
    *,
    gliner_model,
) -> Tuple[dict, List[List[str]]]:
    from gliner.data_processing.collator import TokenDataCollator

    collator = TokenDataCollator(
        gliner_model.config,
        data_processor=gliner_model.data_processor,
        return_tokens=True,
        return_id_to_classes=True,
        prepare_labels=False,
    )
    all_words = [r["tokenized_text"] for r in rows]
    batch = collator(rows, entity_types=labels)
    return batch, all_words


def processor_tokenize_rows(texts: List[str], labels: List[str]) -> List[dict]:
    from plugins.modernbert_gliner_rerank.processor import GLiNERRerankProcessor

    proc = GLiNERRerankProcessor()
    proc._ensure_llm = lambda: None
    proc.warmup(labels)
    return [proc._tokenize(t) for t in texts]


def compare_single(ref: dict, pr: dict) -> bool:
    ok = True
    if ref["words"] != pr["words"]:
        print("FAIL: words list mismatch")
        ok = False
    if ref["text_lengths"] != pr["text_lengths"]:
        print(f"FAIL: text_lengths ref={ref['text_lengths']} pr={pr['text_lengths']}")
        ok = False
    if not torch.equal(ref["input_ids"], pr["input_ids"]):
        print("FAIL: input_ids mismatch")
        ok = False
    if ref["attention_mask"] is not None and pr["attention_mask"] is not None:
        if not torch.equal(ref["attention_mask"], pr["attention_mask"]):
            print("FAIL: attention_mask mismatch")
            ok = False
    elif ref["attention_mask"] != pr["attention_mask"]:
        print("FAIL: attention_mask None mismatch")
        ok = False
    if not torch.equal(ref["words_mask"], pr["words_mask"]):
        print("FAIL: words_mask mismatch")
        ok = False
    return ok


def compare_batched_row(
    batch: dict,
    row: int,
    pr: dict,
    *,
    ref_words: List[str],
) -> bool:
    ids = batch["input_ids"][row].cpu()
    am = batch.get("attention_mask")
    am_row = am[row].cpu() if am is not None else torch.ones_like(ids, dtype=torch.long)
    wm = batch["words_mask"][row].cpu()
    wc = _word_count_row(batch["text_lengths"], row)
    L = _valid_token_len(am_row)
    Lp = _valid_token_len(pr["attention_mask"])

    ok = True
    if ref_words != pr["words"]:
        print(f"FAIL batch row {row}: words list mismatch")
        ok = False
    if wc != pr["text_lengths"]:
        print(f"FAIL batch row {row}: text_lengths ref={wc} pr={pr['text_lengths']}")
        ok = False
    if L != Lp:
        print(f"FAIL batch row {row}: valid token len ref={L} pr={Lp}")
        ok = False
    if not torch.equal(ids[:L], pr["input_ids"]):
        print(f"FAIL batch row {row}: input_ids mismatch (prefix len {L})")
        ok = False
    if not torch.equal(am_row[:L], pr["attention_mask"]):
        print(f"FAIL batch row {row}: attention_mask mismatch (prefix len {L})")
        ok = False
    if not torch.equal(wm[:L], pr["words_mask"]):
        print(f"FAIL batch row {row}: words_mask mismatch (prefix len {L})")
        ok = False
    if L < am_row.numel() and not bool(torch.all(am_row[L:] == 0)):
        print(f"FAIL batch row {row}: expected zero attention_mask past valid length {L}")
        ok = False
    return ok


def run_single(gliner_model) -> bool:
    words = _words_for_text(gliner_model.data_processor, TEST_TEXT)
    batch, wlists = reference_batch_from_rows(
        [{"tokenized_text": words, "ner": None}],
        TEST_LABELS,
        gliner_model=gliner_model,
    )
    am = batch.get("attention_mask")
    tl = batch["text_lengths"]
    tl0 = _word_count_row(tl, 0)
    ref = {
        "input_ids": batch["input_ids"][0].cpu(),
        "attention_mask": am[0].cpu() if am is not None else None,
        "words_mask": batch["words_mask"][0].cpu(),
        "text_lengths": tl0,
        "words": wlists[0],
    }
    pr = processor_tokenize_rows([TEST_TEXT], TEST_LABELS)[0]
    return compare_single(ref, pr)


def run_multi(gliner_model) -> bool:
    rows = []
    for text in MULTI_TEXTS:
        w = _words_for_text(gliner_model.data_processor, text)
        rows.append({"tokenized_text": w, "ner": None})
    batch, wlists = reference_batch_from_rows(rows, TEST_LABELS, gliner_model=gliner_model)
    pr_list = processor_tokenize_rows(MULTI_TEXTS, TEST_LABELS)
    bsz = batch["input_ids"].shape[0]
    if bsz != len(pr_list):
        print(f"FAIL: batch size {bsz} vs processor {len(pr_list)}")
        return False
    ok = True
    for i in range(bsz):
        if not compare_batched_row(batch, i, pr_list[i], ref_words=wlists[i]):
            ok = False
    return ok


def main():
    from gliner import GLiNER

    print("Loading GLiNER (reference collator + words_splitter)...")
    gliner_model = GLiNER.from_pretrained(HF_MODEL_ID)

    print("Single-example parity...")
    if not run_single(gliner_model):
        sys.exit(1)

    print(f"Multi-example batch parity (n={len(MULTI_TEXTS)}, padded collation)...")
    if not run_multi(gliner_model):
        sys.exit(1)

    del gliner_model
    print("PREPROCESS PARITY OK (single + multi batch)")


if __name__ == "__main__":
    main()
