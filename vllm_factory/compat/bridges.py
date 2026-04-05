"""Transport bridges -- translate FactoryRequest into the right vLLM transport.

With vLLM >= 0.19, only the native IOProcessor bridge is used.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from vllm_factory.api.request_models import FactoryRequest
from vllm_factory.api.response_models import FactoryResponse

logger = logging.getLogger("vllm_factory.compat.bridges")


@runtime_checkable
class PoolingBridge(Protocol):
    """Thin contract for request/response translation."""

    def supports_native_io(self) -> bool: ...

    def build_transport_request(self, req: FactoryRequest) -> object: ...

    def parse_transport_response(
        self, resp: object, *, plugin: str, request_id: str | None = None,
    ) -> FactoryResponse: ...


class NativeIOProcessorBridge:
    """Translates via IOProcessorRequest -- zero patching required."""

    def supports_native_io(self) -> bool:
        return True

    def build_transport_request(self, req: FactoryRequest) -> Any:
        try:
            from vllm.entrypoints.pooling.pooling.protocol import (
                IOProcessorRequest,
            )
            return IOProcessorRequest(
                model=req.model,
                data=req.data,
                task="plugin",
            )
        except ImportError:
            raise RuntimeError(
                "IOProcessorRequest not available in this vLLM install"
            )

    def parse_transport_response(
        self, resp: object, *, plugin: str, request_id: str | None = None,
    ) -> FactoryResponse:
        data = getattr(resp, "data", resp)
        return FactoryResponse(
            plugin=plugin,
            request_id=request_id,
            data=data,
        )


def select_bridge(caps: Any | None = None) -> PoolingBridge:
    """Return the native IOProcessor bridge.

    Raises if vLLM does not support the native path.
    """
    if caps is None:
        from vllm_factory.compat.vllm_capabilities import detect
        caps = detect()

    if caps.has_io_processor_interface and caps.has_io_processor_response:
        logger.info("Using native IOProcessor bridge (vLLM %s)", caps.version)
        return NativeIOProcessorBridge()

    raise RuntimeError(
        f"vLLM {caps.version or 'UNKNOWN'} does not support the native "
        "IOProcessor path. vllm-factory requires vLLM >= 0.19."
    )
