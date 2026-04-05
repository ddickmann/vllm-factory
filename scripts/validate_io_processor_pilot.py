#!/usr/bin/env python3
"""
Validate IOProcessor pilot for mmbert_gliner.

Compares three levels:
  1. Structural: Entry-point discovery, class resolution, ABC contract
  2. Pre-processing parity: Same token IDs and metadata from both processor paths
  3. Post-processing parity: Same decoded entities from identical synthetic model output

All tests run without GPU or model weights — they use a public BERT tokenizer
as a surrogate to verify that the IOProcessor and BaseProcessor produce
identical outputs given identical inputs.

Usage:
    python scripts/validate_io_processor_pilot.py [--tokenizer MODEL]
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockGLiNERConfig:
    """Minimal config that satisfies GLiNERPreprocessor and IOProcessor init."""
    max_len = 1024
    max_width = 12
    ent_token = "<<ENT>>"
    sep_token = "<<SEP>>"
    class_token_index = 256000
    sep_token_index = 256001
    model_type = "gliner_mmbert"


class MockModelConfig:
    def __init__(self, tokenizer_name: str):
        self.model = tokenizer_name
        self.hf_config = MockGLiNERConfig()


class MockVllmConfig:
    def __init__(self, tokenizer_name: str):
        self.model_config = MockModelConfig(tokenizer_name)


# ---------------------------------------------------------------------------
# Structural Tests
# ---------------------------------------------------------------------------

def test_entry_point_discovery():
    """Verify the IOProcessor entry-point is discoverable by vLLM."""
    from importlib.metadata import entry_points

    eps = entry_points(group="vllm.io_processor_plugins")
    names = [ep.name for ep in eps]
    assert "mmbert_gliner_io" in names, (
        f"'mmbert_gliner_io' not in discovered entry-points: {names}"
    )

    ep = next(e for e in eps if e.name == "mmbert_gliner_io")
    func = ep.load()
    qualname = func()
    assert qualname == "plugins.mmbert_gliner.io_processor.MMBertGLiNERIOProcessor", (
        f"Unexpected qualname: {qualname}"
    )
    return qualname


def test_class_importable(qualname: str):
    """Verify the IOProcessor class can be resolved."""
    from vllm.utils.import_utils import resolve_obj_by_qualname

    cls = resolve_obj_by_qualname(qualname)
    assert cls.__name__ == "MMBertGLiNERIOProcessor"

    from vllm.plugins.io_processors.interface import IOProcessor
    assert issubclass(cls, IOProcessor), f"{cls} is not a subclass of IOProcessor"
    return cls


def test_abc_contract(cls):
    """Verify all abstract methods are implemented."""
    required = {"parse_request", "pre_process", "post_process", "output_to_response"}
    methods = {m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m))}
    missing = required - methods
    assert not missing, f"Missing abstract methods: {missing}"


# ---------------------------------------------------------------------------
# Pre-processing Parity
# ---------------------------------------------------------------------------

def test_preprocessing_parity(tokenizer_name: str):
    """Compare preprocessing output from Forge BaseProcessor vs IOProcessor.

    Uses a public BERT tokenizer with special tokens added to match GLiNER convention.
    """
    from transformers import AutoTokenizer

    from forge.gliner_preprocessor import GLiNERPreprocessor
    from plugins.mmbert_gliner.io_processor import GLiNERInput, MMBertGLiNERIOProcessor

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<<ENT>>", "<<SEP>>"]})
    config = MockGLiNERConfig()

    preprocessor = GLiNERPreprocessor(
        underlying_tokenizer=tokenizer, config=config,
        device="cpu", include_attention_mask=False,
    )

    text = "The CEO of Apple visited Paris last week."
    labels = ["person", "organization", "location"]

    # --- Forge BaseProcessor path (manual) ---
    result = preprocessor(text, labels, device="cpu")
    enc = result["model_inputs"]
    forge_ids = enc["input_ids"][0].tolist()
    forge_wmask = enc["words_mask"][0].tolist()
    forge_tlen = enc["text_lengths"][0].item()

    # --- IOProcessor path (uses same preprocessor internally) ---
    # Monkey-patch to avoid downloading the model
    io_proc = MMBertGLiNERIOProcessor.__new__(MMBertGLiNERIOProcessor)
    io_proc._tokenizer = tokenizer
    io_proc._preprocessor = GLiNERPreprocessor(
        underlying_tokenizer=tokenizer, config=config,
        device="cpu", include_attention_mask=False,
    )

    import threading
    io_proc._lock = threading.Lock()
    io_proc._pending_extra_kwargs = None
    io_proc._request_meta = {}

    from forge.gliner_postprocessor import GLiNERDecoder
    io_proc._decoder = GLiNERDecoder()

    io_input = GLiNERInput(text=text, labels=labels)
    io_result = io_proc.pre_process(io_input, request_id="test-pre-001")
    io_ids = io_result.get("prompt_token_ids")

    io_params = io_proc.validate_or_generate_params()
    io_extra = io_params.extra_kwargs

    # Compare
    assert forge_ids == io_ids, (
        f"Token IDs differ:\n  Forge: {forge_ids[:20]}...\n  IO:    {io_ids[:20]}..."
    )
    assert forge_wmask == io_extra["words_mask"], "words_mask differs"
    assert forge_tlen == io_extra["text_lengths"], "text_lengths differs"
    assert forge_ids == io_extra["input_ids"], "input_ids in extra_kwargs differs"

    # Verify metadata was stashed for post_process
    assert "test-pre-001" in io_proc._request_meta, "Request metadata not stashed"
    meta = io_proc._request_meta["test-pre-001"]
    assert meta["text"] == text
    assert meta["labels"] == labels
    assert len(meta["tokens"]) > 0
    assert len(meta["word_positions"]) > 0

    # Verify validate_or_generate_params consumed pending state
    assert io_proc._pending_extra_kwargs is None, "Pending extra_kwargs not consumed"

    return {
        "token_ids_match": True,
        "extra_kwargs_match": True,
        "metadata_stashed": True,
        "num_tokens": len(forge_ids),
    }


# ---------------------------------------------------------------------------
# Post-processing Parity
# ---------------------------------------------------------------------------

def test_postprocessing_parity(tokenizer_name: str):
    """Compare post-processing from Forge BaseProcessor vs IOProcessor."""
    from transformers import AutoTokenizer
    from vllm.outputs import PoolingOutput, PoolingRequestOutput

    from forge.gliner_postprocessor import GLiNERDecoder, get_final_entities
    from forge.gliner_preprocessor import GLiNERPreprocessor
    from plugins.mmbert_gliner.io_processor import GLiNERInput, MMBertGLiNERIOProcessor

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<<ENT>>", "<<SEP>>"]})
    config = MockGLiNERConfig()

    preprocessor = GLiNERPreprocessor(
        underlying_tokenizer=tokenizer, config=config,
        device="cpu", include_attention_mask=False,
    )
    decoder = GLiNERDecoder()

    text = "The CEO of Apple visited Paris last week."
    labels = ["person", "organization", "location"]

    # Preprocess to get metadata
    result = preprocessor(text, labels, device="cpu")
    meta = result["postprocessing_metadata"]
    num_words = len(meta["tokens"][0])

    # Create synthetic logits: L words, K=12 max_width, C=3 classes
    L, K, C = num_words, 12, len(labels)
    logits = torch.zeros(L, K, C)
    # "CEO" at word 1, width 0, class 0 (person) — sigmoid(2.0) ≈ 0.88
    logits[1, 0, 0] = 2.0
    # "Apple" at word 3, width 0, class 1 (organization) — sigmoid(2.0) ≈ 0.88
    logits[3, 0, 1] = 2.0
    # "Paris" at word 5, width 0, class 2 (location) — sigmoid(2.0) ≈ 0.88
    logits[5, 0, 2] = 2.0

    shape_prefix = torch.tensor([L, K, C], dtype=logits.dtype)
    raw_output = torch.cat([shape_prefix, logits.flatten()])

    # --- Forge BaseProcessor post-processing (manual) ---
    forge_meta = {
        "text": text,
        "labels": labels,
        "threshold": 0.5,
        "flat_ner": False,
        "multi_label": False,
        "tokens": meta["tokens"],
        "word_positions": meta["word_positions"],
        "id_to_classes": meta["id_to_classes"],
    }

    scores = raw_output.clone()
    fL = int(scores[0].item())
    fK = int(scores[1].item())
    fC = int(scores[2].item())
    forge_logits = scores[3:].reshape(1, fL, fK, fC)

    forge_decoded = decoder.decode(
        tokens=forge_meta["tokens"],
        id_to_classes=forge_meta["id_to_classes"],
        logits=forge_logits,
        flat_ner=False, threshold=0.5, multi_label=False,
    )
    forge_entities = get_final_entities(
        decoded_outputs=forge_decoded,
        word_positions=forge_meta["word_positions"],
        original_texts=[text],
    )[0]

    # --- IOProcessor post-processing ---
    import threading
    io_proc = MMBertGLiNERIOProcessor.__new__(MMBertGLiNERIOProcessor)
    io_proc._tokenizer = tokenizer
    io_proc._preprocessor = preprocessor
    io_proc._decoder = decoder
    io_proc._lock = threading.Lock()
    io_proc._pending_extra_kwargs = None
    io_proc._request_meta = {}

    # Simulate pre_process to stash metadata
    io_input = GLiNERInput(text=text, labels=labels)
    io_proc.pre_process(io_input, request_id="test-post-001")

    # Create mock PoolingRequestOutput
    mock_output = PoolingRequestOutput(
        request_id="test-post-001",
        outputs=PoolingOutput(data=raw_output.tolist()),
        prompt_token_ids=[0] * 10,
        num_cached_tokens=0,
        finished=True,
    )
    io_entities = io_proc.post_process([mock_output], request_id="test-post-001")

    # Compare
    def normalize(ents):
        return sorted(
            [(e["label"], e["text"], round(e["score"], 4)) for e in ents],
            key=lambda x: (x[0], x[1]),
        )

    forge_norm = normalize(forge_entities)
    io_norm = normalize(io_entities)

    assert forge_norm == io_norm, (
        f"Entity lists differ:\n  Forge: {forge_norm}\n  IO:    {io_norm}"
    )

    return {
        "entities_match": True,
        "num_entities": len(forge_entities),
        "entities": forge_norm,
    }


# ---------------------------------------------------------------------------
# IOProcessorResponse contract
# ---------------------------------------------------------------------------

def test_response_contract(tokenizer_name: str):
    """Verify output_to_response returns a valid IOProcessorResponse."""
    from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse

    from plugins.mmbert_gliner.io_processor import MMBertGLiNERIOProcessor

    sample_entities = [
        {"start": 4, "end": 7, "text": "CEO", "label": "person", "score": 0.88},
        {"start": 11, "end": 16, "text": "Apple", "label": "organization", "score": 0.88},
    ]

    # Use output_to_response as a static-like call
    import threading
    io_proc = MMBertGLiNERIOProcessor.__new__(MMBertGLiNERIOProcessor)
    io_proc._lock = threading.Lock()
    io_proc._pending_extra_kwargs = None
    io_proc._request_meta = {}

    response = io_proc.output_to_response(sample_entities)

    assert isinstance(response, IOProcessorResponse), f"Expected IOProcessorResponse, got {type(response)}"
    assert response.data == sample_entities, "Response data doesn't match input"

    return {"response_type": "IOProcessorResponse", "data_preserved": True}


# ---------------------------------------------------------------------------
# validate_or_generate_params contract
# ---------------------------------------------------------------------------

def test_params_contract():
    """Verify validate_or_generate_params correctly stashes/pops extra_kwargs."""
    import threading

    from vllm.pooling_params import PoolingParams

    from plugins.mmbert_gliner.io_processor import MMBertGLiNERIOProcessor

    io_proc = MMBertGLiNERIOProcessor.__new__(MMBertGLiNERIOProcessor)
    io_proc._lock = threading.Lock()
    io_proc._pending_extra_kwargs = None
    io_proc._request_meta = {}

    # Without pending data: returns empty PoolingParams
    params = io_proc.validate_or_generate_params()
    assert isinstance(params, PoolingParams)
    assert params.extra_kwargs is None

    # With pending data: includes extra_kwargs
    test_data = {"input_ids": [1, 2, 3], "words_mask": [0, 1, 0], "text_lengths": 1}
    io_proc._pending_extra_kwargs = test_data
    params = io_proc.validate_or_generate_params()
    assert params.extra_kwargs == test_data

    # Consumed after pop
    assert io_proc._pending_extra_kwargs is None

    # With existing params (offline path)
    io_proc._pending_extra_kwargs = test_data
    existing_params = PoolingParams(task="embed")
    params = io_proc.validate_or_generate_params(existing_params)
    assert params.extra_kwargs == test_data
    assert params is existing_params

    return {"empty_params": True, "pending_data": True, "consumed": True, "merge_existing": True}


# ---------------------------------------------------------------------------
# parse_request contract
# ---------------------------------------------------------------------------

def test_parse_request():
    """Verify parse_request handles various input formats."""
    import threading

    from plugins.mmbert_gliner.io_processor import GLiNERInput, MMBertGLiNERIOProcessor

    io_proc = MMBertGLiNERIOProcessor.__new__(MMBertGLiNERIOProcessor)
    io_proc._lock = threading.Lock()

    # Dict with data key (offline path)
    result = io_proc.parse_request({"data": {"text": "hello", "labels": ["person"]}})
    assert isinstance(result, GLiNERInput)
    assert result.text == "hello"
    assert result.labels == ["person"]
    assert result.threshold == 0.5

    # Object with data attribute (IOProcessorRequest-like)
    class MockReq:
        data = {"text": "world", "labels": ["org"], "threshold": 0.8, "flat_ner": True}
    result = io_proc.parse_request(MockReq())
    assert result.text == "world"
    assert result.threshold == 0.8
    assert result.flat_ner is True

    # Empty labels should raise
    try:
        io_proc.parse_request({"data": {"text": "test", "labels": []}})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "labels" in str(e).lower()

    return {"dict_input": True, "object_input": True, "validation": True}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_test(name, func, *args):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = func(*args)
        elapsed = time.time() - t0
        print(f"  PASS ({elapsed:.2f}s)")
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"    {k}: {v}")
        return True, result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAIL ({elapsed:.2f}s): {e}")
        traceback.print_exc()
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Validate IOProcessor pilot for mmbert_gliner")
    parser.add_argument(
        "--tokenizer", default="bert-base-uncased",
        help="Public tokenizer to use as surrogate (default: bert-base-uncased)",
    )
    args = parser.parse_args()

    results = {}
    all_pass = True

    # Structural tests
    ok, qualname = run_test("Entry-point discovery", test_entry_point_discovery)
    results["entry_point"] = ok
    all_pass &= ok

    if ok:
        ok, cls = run_test("Class importable", test_class_importable, qualname)
        results["class_import"] = ok
        all_pass &= ok

        if ok:
            ok, _ = run_test("ABC contract", test_abc_contract, cls)
            results["abc_contract"] = ok
            all_pass &= ok

    # Contract tests
    ok, _ = run_test("parse_request contract", test_parse_request)
    results["parse_request"] = ok
    all_pass &= ok

    ok, _ = run_test("validate_or_generate_params contract", test_params_contract)
    results["params_contract"] = ok
    all_pass &= ok

    ok, _ = run_test("Response contract", test_response_contract, args.tokenizer)
    results["response_contract"] = ok
    all_pass &= ok

    # Parity tests
    ok, _ = run_test("Pre-processing parity", test_preprocessing_parity, args.tokenizer)
    results["preprocess_parity"] = ok
    all_pass &= ok

    ok, _ = run_test("Post-processing parity", test_postprocessing_parity, args.tokenizer)
    results["postprocess_parity"] = ok
    all_pass &= ok

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print()

    if all_pass:
        print("All tests passed.")
    else:
        print("Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
