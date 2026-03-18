"""
CPU-only: ``batch_predict_entities`` builds one list of variable-length token dicts;
each row matches ``_tokenize`` in isolation (no cross-sample padding).

GPU check: ``scripts/gliner/l4/batch_vllm_parity_test.py`` (sequential vs batched embed).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_L4_SCRIPT_DIR = _REPO_ROOT / "scripts" / "gliner" / "l4"
if str(_L4_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_L4_SCRIPT_DIR))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from l4_parity_fixtures import MULTI_TEXTS, TEST_LABELS  # noqa: E402


def test_multi_text_tokenize_variable_length_and_masks_aligned():
    from plugins.modernbert_gliner_rerank.processor import GLiNERRerankProcessor

    proc = GLiNERRerankProcessor()
    proc._ensure_llm = lambda: None
    proc.warmup(TEST_LABELS)

    per_text = [proc._tokenize(t) for t in MULTI_TEXTS]
    seq_lens = [len(x["input_ids"]) for x in per_text]
    assert len(set(seq_lens)) > 1, "fixture should include different token lengths"

    for i, x in enumerate(per_text):
        L = len(x["input_ids"])
        assert len(x["attention_mask"]) == L, f"row {i} attention_mask vs input_ids"
        assert len(x["words_mask"]) == L, f"row {i} words_mask vs input_ids"
        assert int(x["attention_mask"].sum()) == L, f"row {i}: expect dense mask (no pad tokens in processor path)"
        assert x["text_lengths"] == len(x["words"])


def test_batch_tokenize_matches_collator_rows_like_preprocess_parity():
    """Same invariant as preprocess_parity_test multi path: batched collator row i == _tokenize(texts[i]) on prefix."""
    from gliner import GLiNER
    from gliner.data_processing.collator import TokenDataCollator

    from plugins.modernbert_gliner_rerank.processor import GLiNERRerankProcessor

    gliner_model = GLiNER.from_pretrained("knowledgator/gliner-linker-rerank-v1.0")
    rows = []
    for text in MULTI_TEXTS:
        words = [t for t, _s, _e in gliner_model.data_processor.words_splitter(text)]
        rows.append({"tokenized_text": words, "ner": None})
    collator = TokenDataCollator(
        gliner_model.config,
        data_processor=gliner_model.data_processor,
        return_tokens=True,
        return_id_to_classes=True,
        prepare_labels=False,
    )
    batch = collator(rows, entity_types=TEST_LABELS)
    del gliner_model

    proc = GLiNERRerankProcessor()
    proc._ensure_llm = lambda: None
    proc.warmup(TEST_LABELS)

    import torch

    for i, text in enumerate(MULTI_TEXTS):
        pr = proc._tokenize(text)
        am_row = batch["attention_mask"][i]
        L = int(am_row.sum().item())
        assert torch.equal(batch["input_ids"][i, :L], pr["input_ids"])
        assert torch.equal(am_row[:L], pr["attention_mask"])
        assert torch.equal(batch["words_mask"][i, :L], pr["words_mask"])
