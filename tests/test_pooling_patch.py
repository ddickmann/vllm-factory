from __future__ import annotations

from forge.patches import pooling_extra_kwargs as patch_mod

BASE_PROTOCOL = """
from typing import Generic, TypeAlias, TypeVar
from pydantic import BaseModel, Field

class PoolingCompletionRequest:
    truncate_prompt_tokens: int | None = None
    dimensions: int | None = None
    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            use_activation=get_use_activation(self),
        )


class PoolingChatRequest:
    truncate_prompt_tokens: int | None = None
    dimensions: int | None = None
    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            use_activation=get_use_activation(self),
        )


T = TypeVar("T")
class PoolingResponseData(BaseModel):
    data: list[list[float]] | list[float] | str
"""

ALT_PROTOCOL = """
from __future__ import annotations
from typing import Generic,TypeAlias,TypeVar
from pydantic import BaseModel, Field

class PoolingCompletionRequest(BaseModel):
    truncate_prompt_tokens: int | None = None
    dimensions: int | None = None

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            use_activation=get_use_activation(self),
        )

class PoolingChatRequest(BaseModel):
    truncate_prompt_tokens: int | None = None
    dimensions: int | None = None
    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            use_activation=get_use_activation(self),
        )

T = TypeVar("T")

class PoolingResponseData(BaseModel, Generic[T]):
    index: int
    data: list[list[float]] | list[float] | str
"""


def test_patch_functions_update_protocol_payload_types() -> None:
    patched, changed_extra = patch_mod._patch_extra_kwargs(BASE_PROTOCOL)
    assert changed_extra is True
    assert "extra_kwargs=self.extra_kwargs" in patched

    patched, changed_resp = patch_mod._patch_response_data(patched)
    assert changed_resp is True
    assert "data: Any" in patched


def test_patch_is_idempotent_after_first_application() -> None:
    patched, _ = patch_mod._patch_extra_kwargs(BASE_PROTOCOL)
    patched, _ = patch_mod._patch_response_data(patched)

    patched_again, changed_extra_again = patch_mod._patch_extra_kwargs(patched)
    patched_again, changed_resp_again = patch_mod._patch_response_data(patched_again)
    assert changed_extra_again is False
    assert changed_resp_again is False
    assert patched_again == patched


def test_patch_handles_formatting_variations() -> None:
    content = patch_mod._ensure_any_import(ALT_PROTOCOL)
    assert "from typing import Any" in content

    content, changed_extra = patch_mod._patch_extra_kwargs(content)
    assert changed_extra is True
    assert "extra_kwargs=self.extra_kwargs" in content

    content, changed_resp = patch_mod._patch_response_data(content)
    assert changed_resp is True
    assert "data: Any" in content


def test_version_guard_accepts_015x(monkeypatch) -> None:
    monkeypatch.setattr(patch_mod, "get_installed_vllm_version", lambda: "0.15.7")
    assert patch_mod.ensure_supported_vllm_version(strict=True) is True


def test_version_guard_rejects_outside_range(monkeypatch) -> None:
    monkeypatch.setattr(patch_mod, "get_installed_vllm_version", lambda: "0.16.0")
    monkeypatch.delenv("VLLM_FACTORY_ALLOW_UNSUPPORTED_VLLM", raising=False)
    try:
        patch_mod.ensure_supported_vllm_version(strict=True)
        assert False, "Expected RuntimeError for unsupported version"
    except RuntimeError as exc:
        assert "Unsupported vLLM version" in str(exc)
