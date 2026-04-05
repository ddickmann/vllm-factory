"""Factory-owned response model — insulates callers from vLLM protocol drift."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class FactoryResponse(BaseModel):
    """Stable public response envelope for all vllm-factory plugins.

    IO processors and the legacy bridge both normalise their output into
    this envelope before it reaches callers.
    """

    schema_version: Literal["v1"] = "v1"
    request_id: str | None = None
    plugin: str
    data: Any
    meta: dict[str, Any] | None = None
