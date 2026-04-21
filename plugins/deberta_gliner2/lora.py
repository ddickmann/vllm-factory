"""LoRA request construction helpers for the deberta_gliner2 plugin.

The vLLM engine — not the IOProcessor plugin — selects which LoRA adapter
runs for a given request.  Two call paths are supported upstream:

* **Online (`vllm serve` → `/pooling`)** — `_maybe_get_adapters` matches the
  HTTP ``model`` field against LoRAs registered at startup (``--lora-modules
  name=path``) or via the ``/v1/load_lora_adapter`` endpoint, and sets
  ``ctx.lora_request`` **before** the IOProcessor runs.  The plugin's
  ``adapter`` payload field is a semantic mirror that HTTP shims (e.g. our
  Modal ``/infer`` shim) translate into ``model`` before proxying.

* **Offline (`LLM.encode(...)`)** — the caller passes ``lora_request`` as a
  keyword argument alongside the prompt batch.  vLLM broadcasts a single
  ``LoRARequest`` to every engine input, or zips a list one-to-one.

This module provides the glue for the second path: given a batch of parsed
``GLiNER2Input``s (each with its own ``adapter`` field) and a registry of
``adapter_name -> (int_id, path)``, it returns a list of ``LoRARequest |
None`` ready to hand to ``LLM.encode(..., lora_request=...)``.

The registry is caller-owned on purpose: adapter paths are a deployment
concern (local disk, HF hub, S3 resolver plugin, …) that the plugin has no
business touching.  ``int_id`` must be a positive, deterministic integer per
adapter — vLLM's scheduler uses it as the LoRA identity key when counting
distinct adapters against ``--max-loras``, so two different paths **must
not** share an ``int_id``.

Cross-request LoRA batching falls out of vLLM's v1 scheduler automatically
once each request carries its own ``LoRARequest``: the scheduler tracks
``scheduled_loras: set[int]`` per step (bounded by ``max_loras``) and the
Punica SGMV kernel applies the correct delta per token in a single fused
forward pass.  Nothing in this module needs to model that — we only have to
hand vLLM well-formed ``LoRARequest`` objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest

    from .io_processor import GLiNER2Input


AdapterPath = Union[str, "PathLike[str]"]
AdapterRegistry = Mapping[str, tuple[int, AdapterPath]]
"""Mapping of adapter name to ``(lora_int_id, lora_path)``.

``lora_int_id`` MUST be a unique positive integer per adapter (vLLM uses it
as a cache key in the LoRA model manager and as the identity for
``max_loras`` batching).  ``lora_path`` is any filesystem path or HF repo
identifier that vLLM's LoRA loader understands.
"""


def _resolve_lora_request_cls():
    """Import ``vllm.lora.request.LoRARequest`` lazily.

    vLLM is an optional runtime dep of the plugin's test suite — tests that
    only validate the IOProcessor contract (CPU-only, no GPU) must not force
    a vLLM import.  Production callers importing from this module already
    have vLLM available.
    """
    from vllm.lora.request import LoRARequest  # noqa: PLC0415

    return LoRARequest


def build_lora_request(
    adapter: str | None,
    registry: AdapterRegistry,
    *,
    allow_unknown: bool = False,
) -> "LoRARequest | None":
    """Resolve a single adapter name to a ``LoRARequest``.

    Returns ``None`` when ``adapter`` is ``None`` (base-model request).  When
    the adapter is set but missing from ``registry``, raises ``KeyError``
    unless ``allow_unknown=True`` (in which case the function returns
    ``None`` and the caller gets base-model behaviour — useful for graceful
    fallbacks in benchmark drivers).
    """
    if adapter is None:
        return None
    if adapter not in registry:
        if allow_unknown:
            return None
        raise KeyError(
            f"adapter {adapter!r} not present in registry; known: {sorted(registry.keys())!r}"
        )
    lora_int_id, lora_path = registry[adapter]
    if lora_int_id <= 0:
        raise ValueError(
            f"adapter {adapter!r} has non-positive int_id={lora_int_id}; "
            "vLLM reserves id 0 for the base model"
        )
    lora_request_cls = _resolve_lora_request_cls()
    return lora_request_cls(
        lora_name=adapter,
        lora_int_id=lora_int_id,
        lora_path=str(lora_path),
    )


def build_lora_requests(
    inputs: Sequence["GLiNER2Input"],
    registry: AdapterRegistry,
    *,
    allow_unknown: bool = False,
) -> list["LoRARequest | None"]:
    """Resolve every input's ``adapter`` field in order.

    Intended for offline cross-LoRA bench harnesses that call
    ``llm.encode(batch, lora_request=build_lora_requests(parsed, registry))``.
    The returned list lines up 1:1 with ``inputs`` so vLLM's
    ``_lora_request_to_seq`` length check passes when the engine input count
    matches (our plugin always emits exactly one prompt per input).
    """
    return [
        build_lora_request(inp.adapter, registry, allow_unknown=allow_unknown) for inp in inputs
    ]


__all__ = [
    "AdapterPath",
    "AdapterRegistry",
    "build_lora_request",
    "build_lora_requests",
]
