from __future__ import annotations

import types

import pytest

from forge import preflight


def test_require_pooling_patch_ready_success(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight.pooling_patch, "ensure_supported_vllm_version", lambda strict=True: True
    )
    monkeypatch.setattr(preflight.pooling_patch, "verify_patch", lambda: True)
    monkeypatch.delenv("VLLM_FACTORY_AUTO_APPLY_POOLING_PATCH", raising=False)
    preflight.require_pooling_patch_ready()


def test_require_pooling_patch_ready_raises_when_verify_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight.pooling_patch, "ensure_supported_vllm_version", lambda strict=True: True
    )
    monkeypatch.setattr(preflight.pooling_patch, "verify_patch", lambda: False)
    monkeypatch.delenv("VLLM_FACTORY_AUTO_APPLY_POOLING_PATCH", raising=False)
    with pytest.raises(RuntimeError):
        preflight.require_pooling_patch_ready()


def test_require_pooling_patch_ready_auto_apply(monkeypatch) -> None:
    called = {"apply": False}

    def _apply() -> bool:
        called["apply"] = True
        return True

    monkeypatch.setattr(
        preflight.pooling_patch, "ensure_supported_vllm_version", lambda strict=True: True
    )
    monkeypatch.setattr(preflight.pooling_patch, "apply_patch", _apply)
    monkeypatch.setenv("VLLM_FACTORY_AUTO_APPLY_POOLING_PATCH", "1")
    preflight.require_pooling_patch_ready()
    assert called["apply"] is True


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
