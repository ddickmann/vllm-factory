"""
Async HTTP reverse proxy for multi-instance vLLM serving (beta).

Transparently forwards every incoming HTTP request to one of N vLLM
backend workers, enforcing a per-backend concurrency cap via asyncio
semaphores and selecting backends with round-robin.

The dispatcher is entirely protocol-agnostic: it never inspects request
bodies, does not import any plugin code, and works with any endpoint
(``/pooling``, ``/v1/embeddings``, ``/health``, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

logger = logging.getLogger("vllm-factory.dispatcher")

_DEFAULT_MAX_BS = 32
_CLIENT_TIMEOUT = ClientTimeout(total=300, connect=10)


class Dispatcher:
    """Thin async reverse proxy with per-backend concurrency caps.

    Parameters
    ----------
    backend_urls:
        Base URLs of the vLLM backend servers (e.g. ``["http://127.0.0.1:9100", ...]``).
    max_bs:
        Maximum number of in-flight requests allowed per backend.
        Should match the backend's ``--max-num-seqs``.
    host:
        Host to bind the dispatcher to.
    port:
        Port to bind the dispatcher to.
    """

    def __init__(
        self,
        backend_urls: Sequence[str],
        max_bs: int = _DEFAULT_MAX_BS,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        if not backend_urls:
            raise ValueError("At least one backend URL is required")

        self._backend_urls = list(backend_urls)
        self._max_bs = max(1, max_bs)
        self._host = host
        self._port = port
        self._num_backends = len(self._backend_urls)

        self._semaphores: list[asyncio.Semaphore] = []
        self._rr_counter = 0
        self._session: ClientSession | None = None
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        """Start the dispatcher HTTP server."""
        self._semaphores = [
            asyncio.Semaphore(self._max_bs) for _ in range(self._num_backends)
        ]
        self._rr_counter = 0

        connector = TCPConnector(
            limit=self._max_bs * self._num_backends * 2,
            keepalive_timeout=60,
        )
        self._session = ClientSession(connector=connector, timeout=_CLIENT_TIMEOUT)

        app = web.Application()
        app.router.add_route("*", "/health", self._handle_health)
        app.router.add_route("*", "/{path:.*}", self._handle_proxy)

        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

        logger.info(
            f"Dispatcher listening on {self._host}:{self._port} "
            f"-> {self._num_backends} backends (max_bs={self._max_bs})"
        )

    async def stop(self) -> None:
        """Shut down the dispatcher."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _pick_backend(self) -> int:
        """Select a backend index via round-robin, preferring one with capacity.

        Tries up to N backends starting from the round-robin cursor.
        Falls back to the cursor position if all are at capacity (the
        semaphore will queue the request).
        """
        start = self._rr_counter % self._num_backends
        self._rr_counter += 1

        for offset in range(self._num_backends):
            idx = (start + offset) % self._num_backends
            sem = self._semaphores[idx]
            if sem._value > 0:  # noqa: SLF001 — fast capacity check
                return idx

        return start

    async def _handle_proxy(self, request: web.Request) -> web.Response:
        """Forward a request to a backend and relay the response."""
        assert self._session is not None

        idx = self._pick_backend()
        backend_url = self._backend_urls[idx]
        target = f"{backend_url}/{request.match_info['path']}"
        if request.query_string:
            target = f"{target}?{request.query_string}"

        body = await request.read()

        async with self._semaphores[idx]:
            try:
                async with self._session.request(
                    method=request.method,
                    url=target,
                    headers={
                        k: v
                        for k, v in request.headers.items()
                        if k.lower() not in ("host", "transfer-encoding")
                    },
                    data=body,
                ) as resp:
                    resp_body = await resp.read()
                    return web.Response(
                        status=resp.status,
                        body=resp_body,
                        content_type=resp.content_type,
                    )
            except Exception as exc:
                logger.error(f"Backend {idx} ({backend_url}) error: {exc}")
                return web.Response(
                    status=502,
                    text=f'{{"error": "backend unavailable: {exc}"}}',
                    content_type="application/json",
                )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Aggregate health: healthy if at least one backend is reachable."""
        assert self._session is not None

        for url in self._backend_urls:
            try:
                async with self._session.get(
                    f"{url}/health", timeout=ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        return web.Response(text="OK", content_type="text/plain")
            except Exception:
                continue

        return web.Response(
            status=503,
            text='{"error": "no healthy backends"}',
            content_type="application/json",
        )
