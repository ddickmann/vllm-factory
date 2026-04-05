"""FactoryIOProcessor — the ONE file that touches vLLM's IOProcessor ABC.

When vLLM renames/splits/changes the IOProcessor interface, update THIS file.
All 12 plugin IO processors inherit from :class:`FactoryIOProcessor` and
override only the stable ``factory_*`` methods — they never import vLLM
protocol types directly.

Re-exports
----------
This module also re-exports a few vLLM types that plugins legitimately need
(``TokensPrompt``, ``PromptType``).  When vLLM moves them, update the imports
here — plugin code stays untouched.
"""

from __future__ import annotations

import contextvars
import threading
import uuid
from collections.abc import Sequence
from typing import Any

# ── vLLM imports (centralised — update HERE when vLLM changes) ───────────────
from vllm.config import VllmConfig
from vllm.inputs import TokensPrompt

try:
    from vllm.inputs.data import PromptType
except ImportError:
    from vllm.inputs import PromptType  # type: ignore[attr-defined]

from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams

try:
    from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
except ImportError:
    IOProcessorResponse = None  # type: ignore[assignment,misc]  # 0.18.0+ removed this

# Re-exports for plugins — import from here, not from vllm.*
__all__ = [
    "FactoryIOProcessor",
    "TokensPrompt",
    "PromptType",
    "PoolingRequestOutput",
]


class FactoryIOProcessor(IOProcessor):
    """Stable base class for all vllm-factory IO processors.

    Plugins inherit this and override the ``factory_*`` methods.  This class
    handles:

    * Translation between vLLM's evolving ABC (``parse_data``,
      ``merge_pooling_params``, …) and the stable factory interface.
    * Thread-safe stashing of ``extra_kwargs`` between ``pre_process`` and
      ``merge_pooling_params``.
    * Thread-safe stashing of per-request metadata for ``post_process``.
    * Default ``PoolingParams`` creation (task left as ``None`` so vLLM's
      serving layer resolves it from the model's supported tasks).

    When vLLM renames or reorganises the ABC, update the delegation methods
    below.  The ``factory_*`` signatures stay fixed.
    """

    pooling_task: str | None = None

    def __init__(self, vllm_config: VllmConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        self._extra_lock = threading.Lock()
        self._pending_extra: dict[str, dict[str, Any]] = {}
        self._request_meta: dict[str, Any] = {}
        self._request_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"{self.__class__.__name__}_request_key",
            default=None,
        )

    # ── vLLM ABC delegation (update here when vLLM changes) ──────────────

    def parse_data(self, data: object) -> Any:  # noqa: ANN401
        """vLLM 0.19+ entry point (replaces deprecated ``parse_request``)."""
        return self.factory_parse(data)

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        """vLLM 0.19+ entry point (replaces deprecated
        ``validate_or_generate_params``)."""
        request_key = self._request_key.get()
        with self._extra_lock:
            extra = self._pending_extra.pop(request_key, None) if request_key is not None else None

        task = self.pooling_task
        if params is not None:
            if extra is not None:
                params.extra_kwargs = {**(params.extra_kwargs or {}), **extra}
            if task is not None and params.task is None:
                params.task = task
            return params

        return PoolingParams(task=task, extra_kwargs=extra or {})

    def pre_process(
        self,
        prompt: Any,  # noqa: ANN401
        request_id: str | None = None,
        **kwargs: Any,
    ) -> PromptType | Sequence[PromptType]:
        request_key = request_id or f"_offline:{uuid.uuid4().hex}"
        self._request_key.set(request_key)
        return self.factory_pre_process(prompt, request_id)

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401
        meta_key = request_id or self._request_key.get() or "_offline"
        with self._extra_lock:
            meta = self._request_meta.pop(meta_key, None)
        try:
            return self.factory_post_process(model_output, meta)
        finally:
            self._request_key.set(None)

    # ── vLLM 0.15.1 backward-compat shims ─────────────────────────────────
    # These methods are called by vLLM <= 0.15.x instead of the 0.18+/0.19+
    # equivalents above.  They delegate to the same factory_* methods so
    # plugin code works on all versions without modification.

    def parse_request(self, request: Any) -> Any:  # noqa: ANN401
        """vLLM <= 0.15.x entry point (replaced by ``parse_data`` in 0.18+)."""
        return self.factory_parse(request)

    def validate_or_generate_params(
        self,
        params: Any = None,  # noqa: ANN401  — SamplingParams | PoolingParams | None
    ) -> PoolingParams:
        """vLLM <= 0.15.x entry point (replaced by ``merge_pooling_params`` in 0.18+)."""
        return self.merge_pooling_params(params)

    def output_to_response(self, plugin_output: Any) -> Any:  # noqa: ANN401
        """vLLM <= 0.15.x entry point (removed in 0.18+).

        Wraps plugin output in ``IOProcessorResponse`` when available.
        """
        if IOProcessorResponse is not None:
            return IOProcessorResponse(data=plugin_output)
        return plugin_output

    # ── Stable plugin interface (override these) ─────────────────────────

    def factory_parse(self, data: Any) -> Any:  # noqa: ANN401
        """Parse and validate the incoming request data.

        ``data`` is the raw ``request.data`` dict from the HTTP request
        (or the dict passed to ``llm.encode()`` offline).

        Must return a plugin-specific parsed input object (dataclass, etc.).
        """
        raise NotImplementedError

    def factory_pre_process(
        self,
        parsed_input: Any,  # noqa: ANN401
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        """Convert the parsed input into a vLLM prompt.

        Typically tokenises text and returns ``TokensPrompt(prompt_token_ids=…)``.
        Use :meth:`_stash` to attach ``extra_kwargs`` and/or per-request
        metadata for later retrieval in ``merge_pooling_params`` and
        ``factory_post_process``.
        """
        raise NotImplementedError

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Convert engine output into the final plugin response.

        ``request_meta`` is whatever was stashed via :meth:`_stash` during
        ``factory_pre_process`` (``None`` if nothing was stashed).
        """
        raise NotImplementedError

    # ── Helpers for plugins ───────────────────────────────────────────────

    def _stash(
        self,
        extra_kwargs: dict[str, Any] | None = None,
        request_id: str | None = None,
        meta: Any = None,  # noqa: ANN401
    ) -> None:
        """Thread-safe stash of extra_kwargs and per-request metadata.

        Call this from ``factory_pre_process`` to:
        * Pass ``extra_kwargs`` into ``PoolingParams`` (picked up automatically
          by ``merge_pooling_params``).
        * Store ``meta`` for retrieval in ``factory_post_process``.
        """
        key = request_id or self._request_key.get() or "_offline"
        with self._extra_lock:
            if extra_kwargs is not None:
                self._pending_extra[key] = extra_kwargs
            if meta is not None:
                self._request_meta[key] = meta
