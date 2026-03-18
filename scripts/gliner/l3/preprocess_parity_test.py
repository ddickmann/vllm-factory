#!/usr/bin/env python3
"""
Tensor-level parity: GLiNER BiEncoderTokenDataCollator vs GLiNERLinkerProcessor._tokenize.

Does not load vLLM (monkeypatches _ensure_llm). Requires GPU optional for encode_labels;
runs on CPU if CUDA unavailable (slower).

Usage:
    cd vllm-factory && PYTHONPATH=. python scripts/gliner/l3/preprocess_parity_test.py
"""

from __future__ import annotations

import os
import sys

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_MODEL_ID = "knowledgator/gliner-linker-large-v1.0"

TEST_TEXT = (
    "Farnese Palace is one of the most important palaces in the city of Rome. "
    "Michelangelo also contributed."
)

TEST_LABELS = [
    "Farnese Palace: Renaissance palace",
    "Rome: Capital city",
    "Michelangelo: Italian artist",
]


def reference_batch():
    from gliner import GLiNER
    from gliner.data_processing.collator import BiEncoderTokenDataCollator

    gliner = GLiNER.from_pretrained(HF_MODEL_ID)
    words = []
    for t, _s, _e in gliner.data_processor.words_splitter(TEST_TEXT):
        words.append(t)
    collator = BiEncoderTokenDataCollator(
        gliner.config,
        data_processor=gliner.data_processor,
        return_tokens=True,
        return_id_to_classes=True,
        return_entities=True,
        prepare_labels=False,
    )
    batch = collator(
        [{"tokenized_text": words, "ner": None}],
        entity_types=TEST_LABELS,
    )
    am = batch.get("attention_mask")
    ref = {
        "input_ids": batch["input_ids"][0].cpu(),
        "attention_mask": am[0].cpu() if am is not None else None,
        "words_mask": batch["words_mask"][0].cpu(),
        "text_lengths": batch["text_lengths"][0, 0].item()
        if batch["text_lengths"].dim() == 2
        else batch["text_lengths"][0].item(),
        "words": words,
    }
    del gliner
    return ref


def processor_tokenize():
    from plugins.deberta_gliner_linker.processor import GLiNERLinkerProcessor

    proc = GLiNERLinkerProcessor()
    proc._ensure_llm = lambda: None  # skip vLLM
    proc.warmup(TEST_LABELS)
    tok = proc._tokenize(TEST_TEXT)
    return {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "words_mask": tok["words_mask"],
        "text_lengths": tok["text_lengths"],
        "words": tok["words"],
    }


def main():
    print("Reference collator...")
    ref = reference_batch()
    print("Processor _tokenize...")
    pr = processor_tokenize()

    ok = True
    if ref["words"] != pr["words"]:
        print("FAIL: words list mismatch")
        print("  ref", ref["words"][:20], "...")
        print("  pr ", pr["words"][:20], "...")
        ok = False

    if ref["text_lengths"] != pr["text_lengths"]:
        print(f"FAIL: text_lengths ref={ref['text_lengths']} pr={pr['text_lengths']}")
        ok = False

    if not torch_equal(ref["input_ids"], pr["input_ids"], "input_ids"):
        ok = False
    if ref["attention_mask"] is not None:
        if not torch_equal(ref["attention_mask"], pr["attention_mask"], "attention_mask"):
            ok = False
    if not torch_equal(ref["words_mask"], pr["words_mask"], "words_mask"):
        ok = False

    if ok:
        print("PREPROCESS PARITY OK")
    else:
        print("PREPROCESS PARITY FAILED")
        sys.exit(1)


def torch_equal(a, b, name: str) -> bool:
    if a.shape != b.shape:
        print(f"FAIL: {name} shape {a.shape} vs {b.shape}")
        return False
    if not torch.equal(a, b):
        print(f"FAIL: {name} values differ (max abs { (a - b).abs().max().item() })")
        return False
    return True


if __name__ == "__main__":
    main()
