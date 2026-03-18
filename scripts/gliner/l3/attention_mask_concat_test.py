#!/usr/bin/env python3
"""Unit tests for vLLM pooling attention_mask concatenation (no GPU)."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# Run as: cd vllm-factory && PYTHONPATH=. python scripts/gliner/l3/attention_mask_concat_test.py
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from plugins.deberta_gliner_linker.vllm_pooling_attention_mask import _build_flat_attention_mask


class _FakeInputBatch:
    def __init__(self):
        self.num_reqs = 2
        self.req_ids = ["r0", "r1"]
        self.num_prompt_tokens = np.array([5, 4], dtype=np.int32)
        self.num_computed_tokens_cpu = np.array([0, 0], dtype=np.int32)
        self._params = None

    def get_pooling_params(self):
        return self._params


def test_concat_and_pad():
    batch = _FakeInputBatch()
    batch._params = [
        SimpleNamespace(extra_kwargs={"attention_mask": [1, 1, 1, 0, 0]}),
        SimpleNamespace(extra_kwargs={"attention_mask": [1, 1, 1, 1]}),
    ]
    runner = SimpleNamespace(
        input_batch=batch,
        device=torch.device("cpu"),
    )
    sched = SimpleNamespace(
        num_scheduled_tokens={"r0": 5, "r1": 4},
    )
    out = _build_flat_attention_mask(runner, sched, num_input_tokens=12)
    assert out is not None
    assert out.shape == (12,)
    expected = torch.tensor([1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)
    assert torch.equal(out, expected)


def test_chunked_slice():
    batch = _FakeInputBatch()
    batch.num_computed_tokens_cpu = np.array([2, 0], dtype=np.int32)
    batch._params = [
        SimpleNamespace(extra_kwargs={"attention_mask": [1, 1, 1, 1, 1]}),
        SimpleNamespace(extra_kwargs={"attention_mask": [1, 0, 0, 0]}),
    ]
    runner = SimpleNamespace(input_batch=batch, device=torch.device("cpu"))
    sched = SimpleNamespace(
        num_scheduled_tokens={"r0": 2, "r1": 4},
    )
    out = _build_flat_attention_mask(runner, sched, num_input_tokens=6)
    assert out is not None
    # r0: mask[2:4] -> two ones; r1: full 4 -> 1,0,0,0
    assert torch.equal(out, torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long))


def test_missing_mask_returns_none():
    batch = _FakeInputBatch()
    batch._params = [
        SimpleNamespace(extra_kwargs={"attention_mask": [1, 1, 1, 1, 1]}),
        SimpleNamespace(extra_kwargs={}),
    ]
    runner = SimpleNamespace(input_batch=batch, device=torch.device("cpu"))
    sched = SimpleNamespace(num_scheduled_tokens={"r0": 5, "r1": 4})
    assert _build_flat_attention_mask(runner, sched, 20) is None


def main():
    test_concat_and_pad()
    test_chunked_slice()
    test_missing_mask_returns_none()
    print("attention_mask_concat_test OK")


if __name__ == "__main__":
    main()
