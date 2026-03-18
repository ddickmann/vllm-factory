#!/usr/bin/env python3
"""
GLiNER L4 rerank — entity-level parity: vanilla GLiNER (PyTorch) vs GLiNERRerankProcessor + vLLM.

Uses the same long-form text and KB template as the L3 linker entity test so results are
comparable across stages.

**Status (vLLM 0.15.x, ettin / interleaved sliding attention):** Phase 1 (vanilla
``GLiNER.predict_entities`` on CUDA) is reliable. Phase 2 can fail inside vLLM's
ModernBERT attention (e.g. Triton IMA with float32; FlashAttention / FlexAttention
shape mismatches with float16). Until the engine path is stable, compare
preprocessing with ``scripts/gliner/l4/preprocess_parity_test.py`` (CPU; should print PREPROCESS PARITY OK).

Phases use ``multiprocessing`` ``spawn``. Do not initialize CUDA in the parent before
Phase 2's ``LLM()`` (fork/CUDA re-init issues).

Usage:
    cd vllm-factory && PYTHONPATH=. python scripts/gliner/l4/entity_parity_test.py [--regen]

Pass: same entity count; start/end/text/label match; scores within GLINKER_ENTITY_SCORE_ATOL (default 0.08).
See ``docs/gliner/L4_PARITY.md`` for a short matrix of observed failures.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Avoid loading every vLLM plugin in the child (registry thrash); rerank registers itself on import.
os.environ.setdefault("VLLM_PLUGINS", "modernbert_gliner_rerank")

HF_MODEL_ID = "knowledgator/gliner-linker-rerank-v1.0"
# L4 uni-encoder + vLLM ModernBERT may drift slightly more than L3 bi-encoder.
SCORE_ATOL = float(os.environ.get("GLINKER_ENTITY_SCORE_ATOL", "0.08"))

TEST_TEXT = (
    "Farnese Palace is one of the most important palaces in the city of Rome. "
    "It was designed by Antonio da Sangallo the Younger in 1517 for the Farnese family. "
    "Michelangelo also contributed to its design. "
    "Today it serves as the French Embassy in Italy."
)

TEST_KB = [
    {"entity_id": "Q896027", "label": "Farnese Palace", "description": "Renaissance palace in Rome",
     "entity_type": "building", "aliases": ["Palazzo Farnese"]},
    {"entity_id": "Q220", "label": "Rome", "description": "Capital city of Italy",
     "entity_type": "city", "aliases": ["Roma"]},
    {"entity_id": "Q236", "label": "Antonio da Sangallo the Younger",
     "description": "Italian Renaissance architect",
     "entity_type": "person", "aliases": ["Sangallo"]},
    {"entity_id": "Q5592", "label": "Michelangelo", "description": "Italian sculptor and painter",
     "entity_type": "person", "aliases": ["Michelangelo Buonarroti"]},
    {"entity_id": "Q142", "label": "France", "description": "Country in Western Europe",
     "entity_type": "country", "aliases": ["French Republic"]},
    {"entity_id": "Q38", "label": "Italy", "description": "Country in Southern Europe",
     "entity_type": "country", "aliases": ["Italian Republic"]},
    {"entity_id": "Q78913", "label": "Farnese family", "description": "Italian noble family",
     "entity_type": "family", "aliases": ["House of Farnese"]},
    {"entity_id": "Q123456", "label": "French Embassy", "description": "Diplomatic mission of France",
     "entity_type": "organization", "aliases": []},
]

TEMPLATE = "{label}: {description}"
THRESHOLD = 0.3
REF_FILE = "/tmp/gliner-rerank-entity-reference.json"


def _entity_sort_key(e: dict):
    return (e["start"], e["end"], e["label"], e["text"].lower())


def assert_entity_parity(ref_entities: list, vllm_entities: list) -> None:
    if len(ref_entities) != len(vllm_entities):
        raise AssertionError(
            f"entity count mismatch: ref={len(ref_entities)} vllm={len(vllm_entities)}"
        )

    ref_sorted = sorted(ref_entities, key=_entity_sort_key)
    v_sorted = sorted(vllm_entities, key=_entity_sort_key)

    for i, (re_, ve) in enumerate(zip(ref_sorted, v_sorted)):
        if re_["start"] != ve["start"] or re_["end"] != ve["end"]:
            raise AssertionError(
                f"entity[{i}] span mismatch ref={re_['start']}:{re_['end']} "
                f"vllm={ve['start']}:{ve['end']} text_ref={re_['text']!r} text_vllm={ve['text']!r}"
            )
        if re_["label"] != ve["label"]:
            raise AssertionError(
                f"entity[{i}] label mismatch ref={re_['label']!r} vllm={ve['label']!r}"
            )
        if re_["text"] != ve["text"]:
            raise AssertionError(
                f"entity[{i}] text mismatch ref={re_['text']!r} vllm={ve['text']!r}"
            )
        ds = abs(float(re_["score"]) - float(ve["score"]))
        if ds > SCORE_ATOL:
            raise AssertionError(
                f"entity[{i}] score mismatch ref={re_['score']} vllm={ve['score']} "
                f"(diff={ds}, atol={SCORE_ATOL})"
            )


def _normalize_entities(raw: list) -> list:
    out = []
    for e in raw:
        out.append({
            "text": e["text"],
            "label": e["label"],
            "score": round(float(e["score"]), 4),
            "start": int(e["start"]),
            "end": int(e["end"]),
        })
    return out


def phase_gliner():
    print("=" * 70)
    print("PHASE 1: GLiNER native (PyTorch) — L4 rerank checkpoint")
    print("=" * 70)

    import torch
    from gliner import GLiNER

    if not torch.cuda.is_available():
        print("ERROR: Phase 1 requires CUDA for parity with typical vLLM setup.")
        sys.exit(1)

    labels = [TEMPLATE.format(**e) for e in TEST_KB]
    model = GLiNER.from_pretrained(HF_MODEL_ID)
    model = model.cuda()

    print(f"Text: '{TEST_TEXT[:80]}...'")
    print(f"Labels: {len(labels)}, template='{TEMPLATE}'")
    print(f"Threshold: {THRESHOLD}")

    t0 = time.perf_counter()
    raw = model.predict_entities(
        TEST_TEXT,
        labels,
        flat_ner=True,
        threshold=THRESHOLD,
        multi_label=False,
    )
    latency = (time.perf_counter() - t0) * 1000
    entities = _normalize_entities(raw)

    print(f"\nGLiNER found {len(entities)} entities in {latency:.0f}ms:")
    for e in entities:
        print(f"  '{e['text']}' -> '{e['label'][:48]}' score={e['score']:.4f} [{e['start']}:{e['end']}]")

    with open(REF_FILE, "w") as f:
        json.dump({"entities": entities, "latency_ms": latency}, f, indent=2)
    print(f"\nSaved reference to {REF_FILE}")


def phase_vllm():
    print("=" * 70)
    print("PHASE 2: GLiNERRerankProcessor + vLLM")
    print("=" * 70)

    # Do not call torch.cuda.* before LLM(): vLLM forks an engine subprocess and
    # PyTorch forbids fork after CUDA init in the parent.

    with open(REF_FILE) as f:
        ref = json.load(f)
    ref_entities = ref["entities"]
    print(f"Reference: {len(ref_entities)} entities from GLiNER native")

    from plugins.modernbert_gliner_rerank.processor import GLiNERRerankProcessor

    labels = [TEMPLATE.format(**e) for e in TEST_KB]
    proc = GLiNERRerankProcessor(gpu_memory_utilization=0.45, max_model_len=2048)
    proc.warmup(labels)

    t0 = time.perf_counter()
    vllm_entities = proc.predict_entities(TEST_TEXT, threshold=THRESHOLD)
    latency = (time.perf_counter() - t0) * 1000

    print(f"\nvLLM found {len(vllm_entities)} entities in {latency:.0f}ms:")
    for e in vllm_entities:
        print(f"  '{e['text']}' -> '{e['label'][:48]}' score={e['score']:.4f} [{e['start']}:{e['end']}]")

    print(f"\n{'=' * 70}")
    print("COMPARISON (L4 rerank entity parity)")
    print(f"{'=' * 70}")
    print(f"GLiNER entities: {len(ref_entities)}")
    print(f"vLLM entities:   {len(vllm_entities)}")
    print(f"Score tolerance (abs): {SCORE_ATOL}")

    parity_ok = False
    try:
        assert_entity_parity(ref_entities, vllm_entities)
        parity_ok = True
        print("\nENTITY PARITY OK")
    except AssertionError as err:
        print(f"\nENTITY PARITY FAILED: {err}")

    proc.close()

    if not parity_ok:
        sys.exit(1)


if __name__ == "__main__":
    print(f"Model: {HF_MODEL_ID}")
    print(f"Test: '{TEST_TEXT[:60]}...'")
    print()

    # vLLM v1 forks an engine core; CUDA must not be initialized in a forked parent.
    ctx = mp.get_context("spawn")

    regen = "--regen" in sys.argv

    if regen or not os.path.exists(REF_FILE):
        print("Phase 1: GLiNER reference...\n")
        p1 = ctx.Process(target=phase_gliner)
        p1.start()
        p1.join()
        if p1.exitcode != 0:
            print("Phase 1 FAILED")
            sys.exit(1)
        print()
    else:
        print(f"Using existing reference: {REF_FILE} (pass --regen to regenerate)\n")

    print("Phase 2: vLLM inference...\n")
    p2 = ctx.Process(target=phase_vllm)
    p2.start()
    p2.join()
    sys.exit(p2.exitcode or 0)
