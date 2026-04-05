"""
Multi-instance vLLM server orchestrator (beta).

Launches N identical ``ModelServer`` backends on a single GPU with scaled
GPU-memory budgets, then starts a :class:`~forge.dispatcher.Dispatcher`
reverse proxy in front of them.

This module is the engine behind ``vllm-factory-serve --num-instances N``.
It never modifies existing ``ModelServer`` behaviour — it simply creates
multiple instances with different ports and a smaller memory target.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import List, Optional

from forge.dispatcher import Dispatcher
from forge.server import ModelServer

logger = logging.getLogger("vllm-factory.multi-instance")


def _strip_flag(args: list[str], flag: str) -> list[str]:
    """Remove a CLI flag and its value from an arg list.

    Handles both ``--flag value`` and ``--flag=value`` syntax.
    """
    result: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == flag and i + 1 < len(args):
            i += 2
            continue
        if args[i].startswith(f"{flag}="):
            i += 1
            continue
        result.append(args[i])
        i += 1
    return result


def _scale_gpu_memory(num_instances: int, base: float = 0.92) -> float:
    """Compute per-instance GPU memory utilisation target.

    Leaves a small headroom so that N instances fit comfortably.
    """
    target = base / num_instances
    return max(0.10, round(min(0.80, target), 2))


class MultiInstanceServer:
    """Launch N vLLM backends behind a dispatcher proxy.

    Parameters
    ----------
    model:
        HF model ID or local path.
    num_instances:
        Number of backend vLLM workers.
    max_bs:
        Per-backend concurrency cap (also sets ``--max-num-seqs``).
    port:
        User-facing dispatcher port.
    port_start:
        First internal backend port.  Backends use consecutive ports.
    extra_args:
        Additional CLI flags forwarded to each ``vllm serve`` backend.
        ``--gpu-memory-utilization`` is stripped automatically to avoid
        duplication with the scaled value.
    model_kwargs:
        Keyword arguments forwarded to every ``ModelServer`` constructor
        (e.g. ``dtype``, ``enforce_eager``, ``trust_remote_code``).
    """

    def __init__(
        self,
        model: str,
        num_instances: int = 2,
        max_bs: int = 32,
        port: int = 8000,
        port_start: int = 9100,
        extra_args: Optional[List[str]] = None,
        gpu_memory_utilization: Optional[float] = None,
        **model_kwargs,
    ):
        if num_instances < 2:
            raise ValueError("num_instances must be >= 2 for multi-instance mode")

        self._model = model
        self._num_instances = num_instances
        self._max_bs = max(1, max_bs)
        self._port = port
        self._port_start = port_start
        self._extra_args = _strip_flag(extra_args or [], "--gpu-memory-utilization")
        self._model_kwargs = model_kwargs

        gpu_util = gpu_memory_utilization if gpu_memory_utilization else _scale_gpu_memory(num_instances)
        self._gpu_util = gpu_util

        self._servers: list[ModelServer] = []
        self._dispatcher: Dispatcher | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        for idx in range(num_instances):
            backend_port = port_start + idx
            server = ModelServer(
                name=f"backend-{idx}",
                model=model,
                port=backend_port,
                gpu_memory_utilization=gpu_util,
                max_num_seqs=max_bs,
                extra_args=list(self._extra_args),
                auto_prepare_gliner=idx == 0,
                **model_kwargs,
            )
            self._servers.append(server)

    def start(self) -> None:
        """Start all backends sequentially, then the dispatcher.

        Blocks until interrupted (SIGTERM/SIGINT).
        """
        logger.info(
            f"Starting multi-instance server: {self._num_instances} backends, "
            f"gpu_util={self._gpu_util:.2f}/instance, max_bs={self._max_bs}, "
            f"dispatcher :{self._port}, backends :{self._port_start}-"
            f"{self._port_start + self._num_instances - 1}"
        )

        try:
            for server in self._servers:
                logger.info(f"Launching {server.name} on port {server.port}...")
                server.start()
                logger.info(f"{server.name} ready")
        except Exception:
            logger.error("Backend startup failed, shutting down launched servers")
            self._stop_servers()
            raise

        backend_urls = [
            f"http://127.0.0.1:{self._port_start + i}"
            for i in range(self._num_instances)
        ]
        self._dispatcher = Dispatcher(
            backend_urls=backend_urls,
            max_bs=self._max_bs,
            host="0.0.0.0",
            port=self._port,
        )

        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(self._dispatcher.start())
            logger.info("All backends + dispatcher running. Press Ctrl+C to stop.")
            self._install_signal_handlers()
            self._loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully shut down dispatcher and all backends."""
        if self._dispatcher is not None and self._loop is not None:
            try:
                self._loop.run_until_complete(self._dispatcher.stop())
            except Exception as exc:
                logger.warning(f"Dispatcher shutdown error: {exc}")
            self._dispatcher = None

        if self._loop is not None:
            self._loop.close()
            self._loop = None

        self._stop_servers()

    def _stop_servers(self) -> None:
        for server in reversed(self._servers):
            try:
                server.stop()
            except Exception as exc:
                logger.warning(f"Error stopping {server.name}: {exc}")

    def _install_signal_handlers(self) -> None:
        """Register SIGTERM/SIGINT to trigger a clean shutdown."""
        assert self._loop is not None
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._loop.add_signal_handler(sig, self._loop.stop)
