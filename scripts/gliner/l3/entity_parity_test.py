#!/usr/bin/env python3
"""
GLiNER-Linker Entity-Level Parity Test.

Compares entity output from:
  Phase 1: GLinker library (native PyTorch inference)
  Phase 2: GLiNERLinkerProcessor (GLinker components + vLLM inference)

Both receive the same KB labels + text and should produce similar entities.
The pipeline uses GLinker's tokenizer, decoder, and label encoder — only the
neural forward pass differs (PyTorch vs vLLM).

Each phase runs in a separate subprocess to avoid GPU memory conflicts.

Usage:
    cd vllm-factory && PYTHONPATH=. python scripts/gliner/l3/entity_parity_test.py [--regen]

Pass criteria: same entity count; each entity matches on start, end, text, label;
scores within SCORE_ATOL (default 0.05, override via GLINKER_ENTITY_SCORE_ATOL).
"""

import json
import multiprocessing as mp
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_MODEL_ID = "knowledgator/gliner-linker-large-v1.0"
# Residual gap is mostly vLLM text-encoder numerics vs HF; label-side parity requires
# capping labels_tokenizer.model_max_length like GLinker L3 (see processor._cap_labels_tokenizer_max_length).
SCORE_ATOL = float(os.environ.get("GLINKER_ENTITY_SCORE_ATOL", "0.05"))

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
REF_FILE = "/tmp/glinker-entity-reference.json"


def _entity_sort_key(e: dict):
    return (e["start"], e["end"], e["label"], e["text"].lower())


def assert_entity_parity(ref_entities: list, vllm_entities: list) -> None:
    """Strict L3-style entity list comparison."""
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


# -- Phase 1: GLinker native ---------------------------------------------------

def phase_glinker():
    """Run GLinker library with native PyTorch inference."""
    print("=" * 70)
    print("PHASE 1: GLinker Native (PyTorch)")
    print("=" * 70)

    from glinker import ProcessorFactory

    labels = [TEMPLATE.format(**e) for e in TEST_KB]
    executor = ProcessorFactory.create_simple(
        model_name=HF_MODEL_ID,
        threshold=THRESHOLD,
        template=TEMPLATE,
        device="cuda",
        entities=TEST_KB,
        precompute_embeddings=True,
    )

    print(f"Text: '{TEST_TEXT[:80]}...'")
    print(f"Labels: {len(labels)}, template='{TEMPLATE}'")
    print(f"Threshold: {THRESHOLD}")

    t0 = time.perf_counter()
    result = executor.execute({"texts": [TEST_TEXT]})
    latency = (time.perf_counter() - t0) * 1000

    l3_result = result.get("l3_result")
    entities = []
    if l3_result:
        for text_entities in l3_result.entities:
            if not isinstance(text_entities, list):
                text_entities = [text_entities]
            for ent in text_entities:
                entities.append({
                    "text": ent.text,
                    "label": ent.label,
                    "score": round(ent.score, 4),
                    "start": ent.start,
                    "end": ent.end,
                })

    print(f"\nGLinker found {len(entities)} entities in {latency:.0f}ms:")
    for e in entities:
        print(f"  '{e['text']}' -> '{e['label'][:40]}' "
              f"score={e['score']:.4f} [{e['start']}:{e['end']}]")

    with open(REF_FILE, "w") as f:
        json.dump({"entities": entities, "latency_ms": latency}, f, indent=2)
    print(f"\nSaved reference to {REF_FILE}")


# -- Phase 2: GLinker components + vLLM ----------------------------------------

def phase_vllm():
    """Run GLiNERLinkerProcessor (GLinker components + vLLM inference)."""
    print("=" * 70)
    print("PHASE 2: GLinker + vLLM (GLiNERLinkerProcessor)")
    print("=" * 70)

    with open(REF_FILE) as f:
        ref = json.load(f)
    ref_entities = ref["entities"]
    print(f"Reference: {len(ref_entities)} entities from GLinker native")

    from plugins.deberta_gliner_linker.processor import GLiNERLinkerProcessor

    labels = [TEMPLATE.format(**e) for e in TEST_KB]

    proc = GLiNERLinkerProcessor(gpu_memory_utilization=0.5)
    proc.warmup(labels)

    t0 = time.perf_counter()
    vllm_entities = proc.predict_entities(TEST_TEXT, threshold=THRESHOLD)
    latency = (time.perf_counter() - t0) * 1000

    print(f"\nvLLM found {len(vllm_entities)} entities in {latency:.0f}ms:")
    for e in vllm_entities:
        print(f"  '{e['text']}' -> '{e['label'][:40]}' "
              f"score={e['score']:.4f} [{e['start']}:{e['end']}]")

    # -- Compare (strict) ------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("COMPARISON (strict L3 parity)")
    print(f"{'=' * 70}")
    print(f"GLinker entities: {len(ref_entities)}")
    print(f"vLLM entities:    {len(vllm_entities)}")
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


# -- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Test: '{TEST_TEXT[:60]}...'")
    print(f"KB: {len(TEST_KB)} entities, template='{TEMPLATE}'")
    print()

    regen = "--regen" in sys.argv

    if regen or not os.path.exists(REF_FILE):
        print("Phase 1: GLinker reference...\n")
        p1 = mp.Process(target=phase_glinker)
        p1.start()
        p1.join()
        if p1.exitcode != 0:
            print("Phase 1 FAILED")
            sys.exit(1)
        print()
    else:
        print(f"Using existing reference: {REF_FILE}")
        print("(pass --regen to regenerate)\n")

    print("Phase 2: vLLM inference...\n")
    p2 = mp.Process(target=phase_vllm)
    p2.start()
    p2.join()
    sys.exit(p2.exitcode or 0)
