"""
vLLM Model Server — manages the lifecycle of a vLLM server process.

This is the recommended way to use vLLM Factory plugins: launch a
`vllm serve` process with your plugin installed, then make HTTP
requests to /v1/embeddings or /pooling endpoints.

Adapted from LatenceAI's Superpod server infrastructure.

Usage:
    from forge.server import ModelServer

    server = ModelServer(
        name="colbert",
        model="VAGOsolutions/ModernColBERT",
        port=8000,
    )
    server.start()
    # → curl http://localhost:8000/v1/embeddings
    server.stop()
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

from forge.model_prep import get_gliner_base_model_name, prepare_model_for_vllm_if_needed
from forge.preflight import require_pooling_patch_ready, require_runtime_compatibility

logger = logging.getLogger("vllm-factory.server")


class ModelServer:
    """Launches and manages a single vLLM server process.

    Supports both Unix Domain Sockets (preferred for multi-model pods)
    and TCP localhost (simpler for single-model setups).

    Why server mode?
    - vLLM's server mode is the primary production deployment path
    - Plugins auto-register via entry points on import
    - The /pooling endpoint requires the server (with our patch for extra_kwargs)
    - Server process isolation prevents model loading from blocking callers

    Example:
        server = ModelServer(
            name="gliner",
            model="your-gliner-model",
            port=8001,
            gpu_memory_utilization=0.3,
            trust_remote_code=True,
        )
        server.start()  # Blocks until healthy
        # Now POST to http://localhost:8001/v1/embeddings
    """

    def __init__(
        self,
        name: str,
        model: str,
        socket_path: Optional[str] = None,
        port: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 128,
        max_model_len: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool = False,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        task: Optional[str] = None,
        tokenizer: Optional[str] = None,
        auto_prepare_gliner: bool = True,
        gliner_plugin: Optional[str] = None,
        served_model_name: Optional[str] = None,
        pooler_config: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        startup_timeout: int = 600,
        health_check_interval: float = 2.0,
        cuda_devices: str = "0",
    ):
        self.name = name
        self.model = model
        self.socket_path = socket_path
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.dtype = dtype
        self.quantization = quantization
        self.enforce_eager = enforce_eager
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_chunked_prefill = enable_chunked_prefill
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.task = task
        self.tokenizer = tokenizer
        self.auto_prepare_gliner = auto_prepare_gliner
        self.gliner_plugin = gliner_plugin
        self.served_model_name = served_model_name
        self.pooler_config = pooler_config
        self.extra_args = extra_args or []
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self.cuda_devices = cuda_devices

        self.process: Optional[subprocess.Popen] = None

        # Determine connection mode
        if socket_path:
            self._use_uds = True
        elif port:
            self._use_uds = False
        else:
            # Default to TCP on port 8000
            self.port = 8000
            self._use_uds = False

        logger.info(
            f"[{name}] ModelServer: model={model}, "
            f"{'uds=' + socket_path if self._use_uds else 'port=' + str(self.port)}, "
            f"gpu={gpu_memory_utilization}"
        )

    def _build_command(self) -> List[str]:
        """Build the `vllm serve` command."""
        cmd = ["vllm", "serve", self.model]

        # Connection mode
        if self._use_uds:
            cmd.extend(["--uds", self.socket_path])
        else:
            cmd.extend(["--host", "0.0.0.0", "--port", str(self.port)])

        # Core settings
        cmd.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])
        cmd.extend(["--max-num-seqs", str(self.max_num_seqs)])
        cmd.extend(["--dtype", self.dtype])
        # vLLM 0.15.x serve no longer accepts --task.
        # Keep this option for forward-compatibility and skip empty values.
        if self.task:
            cmd.extend(["--task", self.task])

        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if self.max_num_batched_tokens:
            cmd.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
        if self.quantization and self.quantization.lower() not in ("none", ""):
            cmd.extend(["--quantization", self.quantization])
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        if self.served_model_name:
            cmd.extend(["--served-model-name", self.served_model_name])
        if self.tokenizer:
            cmd.extend(["--tokenizer", self.tokenizer])
        if self.pooler_config:
            cmd.extend(["--pooler-config", self.pooler_config])

        # Prefix caching
        if not self.enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")

        # Chunked prefill
        if not self.enable_chunked_prefill:
            cmd.append("--no-enable-chunked-prefill")

        # Reduce noise
        cmd.extend(["--uvicorn-log-level", "warning"])
        cmd.append("--disable-log-requests")

        # Extra args
        cmd.extend(self.extra_args)

        return cmd

    def _resolve_model_for_server(self) -> None:
        """Prepare GLiNER HF repos into local vLLM-compatible model directories."""
        if not self.auto_prepare_gliner:
            return

        original_model = self.model
        resolved = prepare_model_for_vllm_if_needed(
            model_ref=self.model,
            plugin=self.gliner_plugin,
        )
        if resolved != self.model:
            logger.info(f"[{self.name}] Prepared GLiNER model: {self.model} -> {resolved}")
            self.model = resolved
            # GLiNER translated configs use a custom model_type. Set tokenizer
            # explicitly to the encoder's base model to avoid HF auto-config mapping errors.
            if not self.tokenizer:
                base_tokenizer = get_gliner_base_model_name(original_model)
                if base_tokenizer:
                    self.tokenizer = base_tokenizer
                    logger.info(f"[{self.name}] Using GLiNER base tokenizer: {self.tokenizer}")

    def start(self) -> None:
        """Start the vLLM server process (blocking until healthy)."""
        if self.process is not None and self.process.poll() is None:
            logger.warning(f"[{self.name}] Server already running, skipping start")
            return

        # Enforce patch health for pooling server mode.
        require_pooling_patch_ready()
        require_runtime_compatibility()
        self._resolve_model_for_server()

        cmd = self._build_command()
        logger.info(f"[{self.name}] Starting: {' '.join(cmd)}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        python_bin_dir = os.path.dirname(sys.executable)
        env["PATH"] = f"{python_bin_dir}:{env.get('PATH', '')}"

        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            logger.info(f"[{self.name}] Process started (PID: {self.process.pid})")
            self._wait_for_ready()
            logger.info(f"[{self.name}] Server ready")

            # Start background log reader
            self._start_log_reader()

        except FileNotFoundError:
            raise RuntimeError(
                f"[{self.name}] vllm command not found. Ensure vLLM is installed."
            )
        except Exception as e:
            self.stop()
            raise RuntimeError(f"[{self.name}] Failed to start: {e}")

    def _wait_for_ready(self) -> None:
        """Wait for the server to respond to health checks."""
        import select

        start_time = time.time()
        last_error = None
        last_log_elapsed = 0

        while time.time() - start_time < self.startup_timeout:
            # Read server output (non-blocking)
            if self.process is not None and self.process.stdout is not None:
                try:
                    ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                    if ready:
                        line = self.process.stdout.readline()
                        if line and line.strip():
                            logger.info(f"[{self.name}] {line.rstrip()}")
                except Exception:
                    pass

            # Check if process exited
            if self.process is not None and self.process.poll() is not None:
                if self.process.stdout:
                    remaining = self.process.stdout.read()
                    for line in remaining.split("\n")[-20:]:
                        if line.strip():
                            logger.error(f"[{self.name}] {line}")
                raise RuntimeError(
                    f"[{self.name}] Process exited with code {self.process.returncode}"
                )

            # Health check
            try:
                if self._health_request_sync():
                    elapsed = time.time() - start_time
                    logger.info(f"[{self.name}] Healthy after {elapsed:.1f}s")
                    return
            except Exception as e:
                last_error = e

            # Progress logging
            elapsed = time.time() - start_time
            if elapsed - last_log_elapsed >= 30:
                logger.info(
                    f"[{self.name}] Still starting... "
                    f"({elapsed:.0f}s, last: {last_error})"
                )
                last_log_elapsed = elapsed

            time.sleep(self.health_check_interval)

        raise TimeoutError(
            f"[{self.name}] Failed to start within {self.startup_timeout}s. "
            f"Last error: {last_error}"
        )

    def _start_log_reader(self) -> None:
        """Start a background thread to read vLLM process output."""
        import threading

        def _reader():
            proc = self.process
            if proc is None or proc.stdout is None:
                return
            try:
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        logger.info(f"[{self.name}] {line}")
            except (ValueError, OSError):
                pass

        t = threading.Thread(target=_reader, daemon=True, name=f"log-{self.name}")
        t.start()

    def _health_request_sync(self) -> bool:
        """Synchronous health check."""
        import urllib.request

        if self._use_uds:
            try:
                result = subprocess.run(
                    [
                        "curl",
                        "--unix-socket", self.socket_path,
                        "http://localhost/health",
                        "--max-time", "3",
                        "-sf",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.returncode == 0
            except Exception:
                return False
        else:
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.port}/health", method="GET"
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    return resp.status == 200
            except Exception:
                return False

    async def health_check(self) -> Dict:
        """Async health check."""
        if self.process is None or self.process.poll() is not None:
            return {
                "status": "stopped",
                "name": self.name,
                "pid": None,
                "exit_code": self.process.returncode if self.process else None,
            }

        healthy = self._health_request_sync()
        return {
            "status": "healthy" if healthy else "unhealthy",
            "name": self.name,
            "pid": self.process.pid,
        }

    def stop(self) -> None:
        """Stop the server gracefully."""
        if self.process is None:
            return

        if self.process.poll() is not None:
            logger.debug(
                f"[{self.name}] Already stopped (code {self.process.returncode})"
            )
            self.process = None
            return

        logger.info(f"[{self.name}] Stopping (PID: {self.process.pid})")
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                logger.info(f"[{self.name}] Stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"[{self.name}] Force killing")
                self.process.kill()
                self.process.wait(timeout=5)
        except Exception as e:
            logger.error(f"[{self.name}] Error stopping: {e}")
        finally:
            self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def base_url(self) -> str:
        """Return the base URL for this server."""
        if self._use_uds:
            return "http://localhost"
        return f"http://127.0.0.1:{self.port}"

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        addr = f"uds={self.socket_path}" if self._use_uds else f"port={self.port}"
        return f"ModelServer({self.name!r}, model={self.model!r}, {addr}, {status})"
