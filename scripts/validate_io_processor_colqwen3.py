#!/usr/bin/env python3
"""
Validate IOProcessor for colqwen3 (multimodal ColPali embeddings).

Tests structural correctness and pre/post processing contract parity
against the existing Forge BaseProcessor implementation.

Usage:
    python scripts/validate_io_processor_colqwen3.py
"""

from __future__ import annotations

import sys
import threading
import time
import traceback

import torch

# ---------------------------------------------------------------------------
# Structural Tests
# ---------------------------------------------------------------------------

def test_entry_point_discovery():
    from importlib.metadata import entry_points

    eps = entry_points(group="vllm.io_processor_plugins")
    names = [ep.name for ep in eps]
    assert "colqwen3_io" in names, f"'colqwen3_io' not in: {names}"

    ep = next(e for e in eps if e.name == "colqwen3_io")
    func = ep.load()
    qualname = func()
    assert qualname == "plugins.colqwen3.io_processor.ColQwen3IOProcessor"
    return qualname


def test_class_resolution(qualname: str):
    from vllm.plugins.io_processors.interface import IOProcessor
    from vllm.utils.import_utils import resolve_obj_by_qualname

    cls = resolve_obj_by_qualname(qualname)
    assert cls.__name__ == "ColQwen3IOProcessor"
    assert issubclass(cls, IOProcessor)
    return cls


def test_abc_contract(cls):
    required = {"parse_request", "pre_process", "post_process", "output_to_response"}
    methods = {m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m))}
    missing = required - methods
    assert not missing, f"Missing: {missing}"


# ---------------------------------------------------------------------------
# Contract Tests
# ---------------------------------------------------------------------------

def test_parse_request():
    from plugins.colqwen3.io_processor import ColQwen3Input, ColQwen3IOProcessor

    proc = ColQwen3IOProcessor.__new__(ColQwen3IOProcessor)
    proc._lock = threading.Lock()

    # Text input
    result = proc.parse_request({"data": {"text": "What is ML?"}})
    assert isinstance(result, ColQwen3Input)
    assert result.prompt == "What is ML?"
    assert result.is_query is True

    # Image input
    result = proc.parse_request({"data": {"image": "/path/to/img.png", "is_query": False}})
    assert result.prompt == {"image": "/path/to/img.png"}
    assert result.is_query is False

    # Object with data attr
    class MockReq:
        data = {"text": "hello", "is_query": False}
    result = proc.parse_request(MockReq())
    assert result.prompt == "hello"
    assert result.is_query is False

    # Missing required key
    try:
        proc.parse_request({"data": {"foo": "bar"}})
        assert False, "Should have raised"
    except ValueError:
        pass

    return {"text": True, "image": True, "object": True, "validation": True}


def test_params_contract():
    from vllm.pooling_params import PoolingParams

    from plugins.colqwen3.io_processor import ColQwen3IOProcessor

    proc = ColQwen3IOProcessor.__new__(ColQwen3IOProcessor)
    proc._lock = threading.Lock()
    proc._pending_extra_kwargs = None

    # Without pending data
    params = proc.validate_or_generate_params()
    assert isinstance(params, PoolingParams)
    assert params.task == "token_embed"

    # With pending data
    proc._pending_extra_kwargs = {"is_query": True}
    params = proc.validate_or_generate_params()
    assert params.extra_kwargs == {"is_query": True}
    assert params.task == "token_embed"
    assert proc._pending_extra_kwargs is None

    return {"default_task": True, "extra_kwargs": True, "consumed": True}


# ---------------------------------------------------------------------------
# Pre/Post Processing Parity
# ---------------------------------------------------------------------------

def test_preprocessing_parity():
    """Compare Forge BaseProcessor preprocess with IOProcessor pre_process."""

    from plugins.colqwen3.io_processor import ColQwen3Input, ColQwen3IOProcessor
    from plugins.colqwen3.processor import ColQwen3Processor

    text = "What is machine learning?"

    # --- Forge path (manual, no engine) ---
    forge_proc = ColQwen3Processor.__new__(ColQwen3Processor)
    forge_result = forge_proc.preprocess(text, is_query=True)
    forge_prompt = forge_result.prompt
    forge_task = forge_result.pooling_params.task
    forge_extra = forge_result.pooling_params.extra_kwargs

    # --- IOProcessor path ---
    io_proc = ColQwen3IOProcessor.__new__(ColQwen3IOProcessor)
    io_proc._lock = threading.Lock()
    io_proc._pending_extra_kwargs = None

    io_input = ColQwen3Input(prompt=text, is_query=True)
    io_prompt = io_proc.pre_process(io_input, request_id="test-001")
    io_params = io_proc.validate_or_generate_params()

    # Compare
    assert forge_prompt == io_prompt, f"Prompts differ: {forge_prompt!r} vs {io_prompt!r}"
    assert forge_task == io_params.task, f"Tasks differ: {forge_task} vs {io_params.task}"
    assert forge_extra == io_params.extra_kwargs, (
        f"Extra kwargs differ: {forge_extra} vs {io_params.extra_kwargs}"
    )

    # Image input
    img_data = {"image": "/path/to/test.png"}
    forge_result = forge_proc.preprocess(img_data, is_query=False)
    io_input = ColQwen3Input(prompt=img_data, is_query=False)
    io_prompt = io_proc.pre_process(io_input, request_id="test-002")
    io_params = io_proc.validate_or_generate_params()

    assert forge_result.prompt == io_prompt
    assert forge_result.pooling_params.extra_kwargs == io_params.extra_kwargs

    return {"text_parity": True, "image_parity": True, "task_match": True}


def test_postprocessing_parity():
    """Compare Forge postprocess with IOProcessor post_process."""
    from vllm.outputs import PoolingOutput, PoolingRequestOutput

    from plugins.colqwen3.io_processor import ColQwen3IOProcessor
    from plugins.colqwen3.processor import ColQwen3Processor

    # Synthetic 2D embeddings (8 tokens × 128 dims)
    raw = torch.randn(8, 128).tolist()

    # --- Forge path ---
    forge_proc = ColQwen3Processor.__new__(ColQwen3Processor)
    forge_result = forge_proc.postprocess(raw)
    forge_flat = forge_result.flatten().tolist()

    # --- IOProcessor path ---
    io_proc = ColQwen3IOProcessor.__new__(ColQwen3IOProcessor)
    io_proc._lock = threading.Lock()

    mock_output = PoolingRequestOutput(
        request_id="test-post-001",
        outputs=PoolingOutput(data=raw),
        prompt_token_ids=[0] * 10,
        num_cached_tokens=0,
        finished=True,
    )
    io_result = io_proc.post_process([mock_output], request_id="test-post-001")

    # io_result is a flat list, forge_result is a tensor
    io_flat = [float(x) for row in io_result for x in (row if isinstance(row, list) else [row])]

    assert len(forge_flat) == len(io_flat), f"Length mismatch: {len(forge_flat)} vs {len(io_flat)}"
    for i, (f, io) in enumerate(zip(forge_flat, io_flat)):
        assert abs(f - io) < 1e-6, f"Value mismatch at {i}: {f} vs {io}"

    return {"embedding_parity": True, "length": len(forge_flat)}


def test_response_contract():
    from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse

    from plugins.colqwen3.io_processor import ColQwen3IOProcessor

    proc = ColQwen3IOProcessor.__new__(ColQwen3IOProcessor)

    sample = [[0.1, 0.2], [0.3, 0.4]]
    response = proc.output_to_response(sample)

    assert isinstance(response, IOProcessorResponse)
    assert response.data == sample

    return {"response_type": "IOProcessorResponse", "data_preserved": True}


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
    results = {}
    all_pass = True

    ok, qualname = run_test("Entry-point discovery", test_entry_point_discovery)
    results["entry_point"] = ok
    all_pass &= ok

    if ok:
        ok, cls = run_test("Class resolution", test_class_resolution, qualname)
        results["class_resolution"] = ok
        all_pass &= ok

        if ok:
            ok, _ = run_test("ABC contract", test_abc_contract, cls)
            results["abc_contract"] = ok
            all_pass &= ok

    ok, _ = run_test("parse_request contract", test_parse_request)
    results["parse_request"] = ok
    all_pass &= ok

    ok, _ = run_test("Params contract", test_params_contract)
    results["params_contract"] = ok
    all_pass &= ok

    ok, _ = run_test("Pre-processing parity", test_preprocessing_parity)
    results["preprocess_parity"] = ok
    all_pass &= ok

    ok, _ = run_test("Post-processing parity", test_postprocessing_parity)
    results["postprocess_parity"] = ok
    all_pass &= ok

    ok, _ = run_test("Response contract", test_response_contract)
    results["response_contract"] = ok
    all_pass &= ok

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
