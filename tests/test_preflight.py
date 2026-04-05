from __future__ import annotations

import types

import pytest

from forge import preflight


def test_require_runtime_compatibility_skips_on_cuda(monkeypatch) -> None:
    fake_cuda = types.SimpleNamespace(is_available=lambda: True)
    fake_torch = types.SimpleNamespace(cuda=fake_cuda, ops=types.SimpleNamespace())
    monkeypatch.delenv("VLLM_FACTORY_SKIP_RUNTIME_COMPAT_CHECK", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    preflight.require_runtime_compatibility()


def test_require_runtime_compatibility_raises_cpu_only_missing_op(monkeypatch) -> None:
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch = types.SimpleNamespace(
        cuda=fake_cuda,
        ops=types.SimpleNamespace(_C_utils=types.SimpleNamespace()),
    )
    monkeypatch.delenv("VLLM_FACTORY_SKIP_RUNTIME_COMPAT_CHECK", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    with pytest.raises(RuntimeError):
        preflight.require_runtime_compatibility()


def test_require_runtime_compatibility_passes_cpu_with_op(monkeypatch) -> None:
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    c_utils = types.SimpleNamespace(init_cpu_threads_env=lambda *_: True)
    fake_torch = types.SimpleNamespace(
        cuda=fake_cuda,
        ops=types.SimpleNamespace(_C_utils=c_utils),
    )
    monkeypatch.delenv("VLLM_FACTORY_SKIP_RUNTIME_COMPAT_CHECK", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    preflight.require_runtime_compatibility()
