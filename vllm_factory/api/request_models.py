"""Factory-owned request model — insulates callers from vLLM protocol drift."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class FactoryRequest(BaseModel):
    """Stable public request envelope for all vllm-factory plugins.

    Translating this into the correct vLLM transport (IOProcessorRequest on
    native installs, patched PoolingCompletionRequest on legacy) is the job
    of ``vllm_factory.compat.bridges``.
    """

    schema_version: Literal["v1"] = "v1"
    model: str
    plugin: str | None = None
    data: dict[str, Any]
    options: dict[str, Any] | None = None
