#!/usr/bin/env python3
"""
Variable-length **multi-request** vLLM parity for L4 rerank.

What happens when several texts of different lengths are passed to
``batch_predict_entities``?

1. **No single padded batch tensor in our code.** Each string is tokenized with
   ``_tokenize`` on its own, so ``input_ids`` / ``attention_mask`` lengths differ per
   item (no cross-sample padding in the processor).

2. **One ``LLM.embed`` call** with a list of ``TokensPrompt`` objects (one per text) and
   matching ``PoolingParams.extra_kwargs`` (masks, ``words_mask``, ``text_lengths``).

3. **Inside vLLM v1**, the scheduler may run multiple requests in one engine step. The
   pooling ``attention_mask`` patch (``vllm_pooling_attention_mask``) builds a **flat**
   mask by concatenating each request's slice of ``extra_kwargs["attention_mask"]`` in
   scheduler order — it does **not** assume one rectangular ``[B, T]`` batch.

This script checks that **batched** ``embed`` (all texts in one call) produces the same
decoded entities as **sequential** ``predict_entities`` (one embed per text), using the
same labels and threshold. That catches mask/ordering bugs when lengths differ.

Requires CUDA + working vLLM for this model (see ``docs/gliner/L4_PARITY.md`` if the engine crashes).

Usage::

    cd vllm-factory && PYTHONPATH=. python scripts/gliner/l4/batch_vllm_parity_test.py

Uses ``multiprocessing`` ``spawn`` so the parent never initializes CUDA before vLLM starts.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_PLUGINS", "modernbert_gliner_rerank")

from l4_parity_fixtures import MULTI_TEXTS, TEST_LABELS

THRESHOLD = 0.35
SCORE_ATOL = float(os.environ.get("GLINKER_ENTITY_SCORE_ATOL", "0.08"))


def _entity_sort_key(e: dict):
    return (e["start"], e["end"], e["label"], e["text"].lower())


def _entities_close(a: dict, b: dict) -> tuple[bool, str]:
    if a["start"] != b["start"] or a["end"] != b["end"]:
        return False, f"span {a} vs {b}"
    if a["label"] != b["label"]:
        return False, f"label {a['label']!r} vs {b['label']!r}"
    if a["text"] != b["text"]:
        return False, f"text {a['text']!r} vs {b['text']!r}"
    if abs(float(a["score"]) - float(b["score"])) > SCORE_ATOL:
        return False, f"score {a['score']} vs {b['score']}"
    return True, ""


def _compare_entity_lists(name_a: str, list_a: list, name_b: str, list_b: list, idx: int) -> None:
    if len(list_a) != len(list_b):
        raise AssertionError(
            f"text[{idx}] entity count {name_a}={len(list_a)} {name_b}={len(list_b)}"
        )
    sa = sorted(list_a, key=_entity_sort_key)
    sb = sorted(list_b, key=_entity_sort_key)
    for j, (x, y) in enumerate(zip(sa, sb)):
        ok, msg = _entities_close(x, y)
        if not ok:
            raise AssertionError(f"text[{idx}] entity[{j}] {msg}")


def run_gpu_child():
    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available (batch vLLM test needs GPU).")
        sys.exit(2)

    from plugins.modernbert_gliner_rerank.processor import GLiNERRerankProcessor

    texts = list(MULTI_TEXTS)
    labels = list(TEST_LABELS)

    print("=" * 70)
    print("BATCH vs SEQUENTIAL vLLM (variable-length texts)")
    print("=" * 70)
    print(f"Texts: {len(texts)} (token lengths differ after _tokenize)")
    for i, t in enumerate(texts):
        print(f"  [{i}] len={len(t)!r} chars")
    print(f"Labels: {len(labels)}, threshold={THRESHOLD}, score_atol={SCORE_ATOL}")
    print()

    proc = None
    try:
        proc = GLiNERRerankProcessor(gpu_memory_utilization=0.45, max_model_len=2048)
        proc.warmup(labels)

        print("Sequential predict_entities (one embed per text)...")
        sequential: list = []
        for i, t in enumerate(texts):
            sequential.append(proc.predict_entities(t, threshold=THRESHOLD))
            print(f"  [{i}] {len(sequential[-1])} entities")

        print("batch_predict_entities (one embed() with N prompts)...")
        batched = proc.batch_predict_entities(texts, threshold=THRESHOLD)
        for i, ents in enumerate(batched):
            print(f"  [{i}] {len(ents)} entities")

        print()
        print("Comparing lists...")
        for i in range(len(texts)):
            _compare_entity_lists("sequential", sequential[i], "batched", batched[i], i)

        print("BATCH vLLM PARITY OK (sequential == batched per text)")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        if proc is not None:
            proc.close()
    sys.exit(0)


def main():
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=run_gpu_child)
    p.start()
    p.join()
    code = p.exitcode
    if code == 2:
        sys.exit(0)  # skip without failure
    sys.exit(code if code is not None else 1)


if __name__ == "__main__":
    main()
