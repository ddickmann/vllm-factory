#!/usr/bin/env python3
"""
Experimental GLiNER2 multi-server benchmark.

Runs a vLLM-only throughput experiment for GLiNER2 with multiple server
instances sharing a single GPU. Requests are dispatched with strict
round-robin assignment across identical backends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.registry import get_entry

DEFAULT_SERVER_COUNTS = (1, 2, 4, 6)
DEFAULT_PER_SERVER_CONCURRENCY = 32
DEFAULT_NUM_REQUESTS = 500
DEFAULT_WARMUP_PER_SERVER = 16
DEFAULT_PORT_START = 9998
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bench" / "results_multi_server"


@dataclass
class MultiServerTopologyResult:
    server_count: int
    per_server_concurrency_cap: int
    global_concurrency: int
    num_requests: int
    ports: list[int]
    status: str
    req_per_s: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    per_server_request_counts: list[int] = field(default_factory=list)
    per_server_mean_latency_ms: list[float] = field(default_factory=list)
    gpu_memory_utilization_per_server: float = 0.0
    error: str | None = None
    log_paths: list[str] = field(default_factory=list)


@dataclass
class MultiServerExperimentResult:
    plugin: str
    model_id: str
    served_model_id: str
    gpu: str
    seq_len: int
    num_requests: int
    server_counts: list[int]
    per_server_concurrency_cap: int
    warmup_per_server: int
    dataset_label: str
    results: list[MultiServerTopologyResult]
    best_server_count: int | None
    best_req_per_s: float | None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dtype: str = "bfloat16"
    vllm_version: str = ""
    compat_mode: str = "native"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        gpu_slug = _slugify(self.gpu)
        model_slug = _slugify(self.model_id)
        path = output_dir / f"{self.plugin}_multi_server_{model_slug}_{gpu_slug}_{ts}.json"
        path.write_text(self.to_json())
        return path


def _slugify(value: str) -> str:
    allowed = []
    for ch in value.lower():
        if ch.isalnum():
            allowed.append(ch)
        else:
            allowed.append("_")
    slug = "".join(allowed).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _detect_gpu() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def _detect_vllm_version() -> str:
    try:
        from vllm_factory.compat.vllm_capabilities import detect

        caps = detect()
        return caps.version or "unknown"
    except Exception:
        return "unknown"


def _latency_summary(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "mean_ms": 0.0,
        }

    latencies = sorted(latencies)
    n = len(latencies)
    return {
        "p50_ms": latencies[min(n - 1, int(n * 0.50))],
        "p95_ms": latencies[min(n - 1, int(n * 0.95))],
        "p99_ms": latencies[min(n - 1, int(n * 0.99))],
        "mean_ms": statistics.mean(latencies),
    }


def _replace_flag_value(flags: list[str], flag_name: str, new_value: str) -> list[str]:
    updated: list[str] = []
    i = 0
    replaced = False
    while i < len(flags):
        flag = flags[i]
        if flag == flag_name and i + 1 < len(flags):
            updated.extend([flag_name, new_value])
            replaced = True
            i += 2
            continue
        updated.append(flag)
        i += 1
    if not replaced:
        updated.extend([flag_name, new_value])
    return updated


def _scaled_serve_flags(base_flags: list[str], server_count: int) -> tuple[list[str], float]:
    if server_count <= 0:
        raise ValueError("server_count must be positive")

    target_util = min(0.80, 0.92 / server_count)
    target_util = max(0.12, round(target_util, 2))
    scaled = _replace_flag_value(base_flags, "--gpu-memory-utilization", f"{target_util:.2f}")
    return scaled, target_util


def _start_server(
    entry,
    port: int,
    serve_flags: list[str],
    per_server_concurrency: int,
    log_path: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        entry.model_id,
        "--io-processor-plugin",
        entry.io_plugin,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--max-num-seqs",
        str(per_server_concurrency),
    ]
    cmd += serve_flags

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    env = os.environ.copy()
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        start_new_session=True,
    )
    log_fh.close()
    return proc


def _wait_healthy(port: int, timeout_s: int = 240) -> bool:
    import requests as req

    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout_s:
        try:
            resp = req.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except req.ConnectionError:
            pass
        time.sleep(2)
    return False


def _kill_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5)
    time.sleep(2)


def _read_log_tail(path: Path, max_chars: int = 3000) -> str:
    if not path.exists():
        return "(log file missing)"
    return path.read_text(errors="replace")[-max_chars:]


def _start_topology(entry, server_count: int, per_server_concurrency: int, port_start: int, log_dir: Path):
    serve_flags, gpu_mem_util = _scaled_serve_flags(entry.serve_flags, server_count)
    procs: list[subprocess.Popen] = []
    ports: list[int] = []
    log_paths: list[Path] = []
    timestamp = int(time.time())
    try:
        for idx in range(server_count):
            port = port_start + idx
            log_path = log_dir / f"{entry.plugin_name}_{server_count}srv_p{port}_{timestamp}.log"
            proc = _start_server(
                entry,
                port=port,
                serve_flags=serve_flags,
                per_server_concurrency=per_server_concurrency,
                log_path=log_path,
            )
            procs.append(proc)
            ports.append(port)
            log_paths.append(log_path)
            if not _wait_healthy(port):
                tail = _read_log_tail(log_path)
                raise RuntimeError(f"server on port {port} failed health check\n{tail}")

        for port, log_path in zip(ports, log_paths):
            if not _wait_healthy(port, timeout_s=15):
                tail = _read_log_tail(log_path)
                raise RuntimeError(f"server on port {port} became unhealthy\n{tail}")

        return procs, ports, log_paths, gpu_mem_util
    except Exception:
        for proc in procs:
            _kill_server(proc)
        raise


async def _send_request(
    session: aiohttp.ClientSession,
    url: str,
    entry,
    payload_data,
) -> float:
    body = {
        "model": entry.model_id,
        entry.payload_key: payload_data,
    }
    if entry.request_task is not None:
        body["task"] = entry.request_task

    start = time.perf_counter()
    async with session.post(url, json=body) as resp:
        raw = await resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} from {url}: {raw[:300]}")
        if raw[:10].startswith(b'{"error"'):
            raise RuntimeError(f"Server returned error from {url}: {raw[:300]}")
    return (time.perf_counter() - start) * 1000


async def _warmup_servers(
    entry,
    dataset: list,
    urls: list[str],
    per_server_concurrency: int,
    warmup_per_server: int,
) -> None:
    total_requests = len(urls) * warmup_per_server
    semaphores = [asyncio.Semaphore(per_server_concurrency) for _ in urls]
    connector = aiohttp.TCPConnector(limit=max(16, len(urls) * per_server_concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        async def warm_one(request_idx: int) -> None:
            server_idx = request_idx % len(urls)
            async with semaphores[server_idx]:
                await _send_request(
                    session=session,
                    url=f"{urls[server_idx]}{entry.endpoint}",
                    entry=entry,
                    payload_data=dataset[request_idx % len(dataset)],
                )

        await asyncio.gather(*(warm_one(i) for i in range(total_requests)))


async def _run_round_robin_saturate(
    entry,
    dataset: list,
    urls: list[str],
    num_requests: int,
    per_server_concurrency: int,
) -> dict:
    semaphores = [asyncio.Semaphore(per_server_concurrency) for _ in urls]
    request_counts = [0 for _ in urls]
    per_server_latencies: list[list[float]] = [[] for _ in urls]
    connector = aiohttp.TCPConnector(limit=max(16, len(urls) * per_server_concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.perf_counter()

        async def run_one(request_idx: int) -> float:
            server_idx = request_idx % len(urls)
            async with semaphores[server_idx]:
                latency = await _send_request(
                    session=session,
                    url=f"{urls[server_idx]}{entry.endpoint}",
                    entry=entry,
                    payload_data=dataset[request_idx % len(dataset)],
                )
            request_counts[server_idx] += 1
            per_server_latencies[server_idx].append(latency)
            return latency

        latencies = await asyncio.gather(*(run_one(i) for i in range(num_requests)))
        elapsed = time.perf_counter() - start

    summary = _latency_summary(latencies)
    return {
        "req_per_s": num_requests / elapsed if elapsed > 0 else 0.0,
        "per_server_request_counts": request_counts,
        "per_server_mean_latency_ms": [
            round(statistics.mean(server_latencies), 2) if server_latencies else 0.0
            for server_latencies in per_server_latencies
        ],
        **summary,
    }


def _best_result(results: list[MultiServerTopologyResult]) -> MultiServerTopologyResult | None:
    successful = [result for result in results if result.status == "ok"]
    if not successful:
        return None
    return min(successful, key=lambda result: (-result.req_per_s, result.p95_ms, result.server_count))


def run_experiment(
    plugin_name: str,
    server_counts: list[int],
    per_server_concurrency: int,
    num_requests: int,
    warmup_per_server: int,
    port_start: int,
    output_dir: Path,
) -> MultiServerExperimentResult:
    entry = get_entry(plugin_name)
    if entry.prep_fn is not None:
        print("[Phase 0] Preparing model...")
        entry.prep_fn()

    dataset = entry.get_dataset()
    gpu = _detect_gpu()
    vllm_version = _detect_vllm_version()
    topology_results: list[MultiServerTopologyResult] = []
    log_dir = output_dir / "server_logs"

    print("=" * 72)
    print("  GLiNER2 Multi-Server Benchmark")
    print("=" * 72)
    print(f"  Plugin:                 {plugin_name}")
    print(f"  Model:                  {entry.model_id}")
    print(f"  GPU:                    {gpu}")
    print(f"  vLLM:                   {vllm_version}")
    print(f"  Num requests:           {num_requests}")
    print(f"  Per-server concurrency: {per_server_concurrency}")
    print(f"  Warmup/server:          {warmup_per_server}")
    print(f"  Topologies:             {server_counts}")
    print("=" * 72)

    for server_count in server_counts:
        global_concurrency = server_count * per_server_concurrency
        print(f"\n[Topology] {server_count} server(s) -> {global_concurrency} max in-flight requests")
        procs: list[subprocess.Popen] = []
        ports: list[int] = []
        log_paths: list[Path] = []
        gpu_mem_util = 0.0
        try:
            procs, ports, log_paths, gpu_mem_util = _start_topology(
                entry=entry,
                server_count=server_count,
                per_server_concurrency=per_server_concurrency,
                port_start=port_start,
                log_dir=log_dir,
            )
            urls = [f"http://127.0.0.1:{port}" for port in ports]
            print(f"  Ports:                  {ports}")
            print(f"  GPU mem/server target:  {gpu_mem_util:.2f}")
            print("  Warmup...")
            asyncio.run(
                _warmup_servers(
                    entry=entry,
                    dataset=dataset,
                    urls=urls,
                    per_server_concurrency=per_server_concurrency,
                    warmup_per_server=warmup_per_server,
                )
            )
            print("  Timed run...")
            metrics = asyncio.run(
                _run_round_robin_saturate(
                    entry=entry,
                    dataset=dataset,
                    urls=urls,
                    num_requests=num_requests,
                    per_server_concurrency=per_server_concurrency,
                )
            )
            result = MultiServerTopologyResult(
                server_count=server_count,
                per_server_concurrency_cap=per_server_concurrency,
                global_concurrency=global_concurrency,
                num_requests=num_requests,
                ports=ports,
                status="ok",
                req_per_s=round(metrics["req_per_s"], 2),
                p50_ms=round(metrics["p50_ms"], 2),
                p95_ms=round(metrics["p95_ms"], 2),
                p99_ms=round(metrics["p99_ms"], 2),
                mean_ms=round(metrics["mean_ms"], 2),
                per_server_request_counts=metrics["per_server_request_counts"],
                per_server_mean_latency_ms=metrics["per_server_mean_latency_ms"],
                gpu_memory_utilization_per_server=gpu_mem_util,
                log_paths=[str(path) for path in log_paths],
            )
            topology_results.append(result)
            print(
                "  Result:"
                f" req/s={result.req_per_s:.2f}"
                f" p50={result.p50_ms:.2f}ms"
                f" p95={result.p95_ms:.2f}ms"
                f" p99={result.p99_ms:.2f}ms"
            )
            print(f"  Per-server request counts: {result.per_server_request_counts}")
        except Exception as exc:
            error_message = str(exc)
            result = MultiServerTopologyResult(
                server_count=server_count,
                per_server_concurrency_cap=per_server_concurrency,
                global_concurrency=global_concurrency,
                num_requests=num_requests,
                ports=ports,
                status="failed",
                error=error_message,
                gpu_memory_utilization_per_server=gpu_mem_util,
                log_paths=[str(path) for path in log_paths],
            )
            topology_results.append(result)
            print(f"  FAILED: {error_message}")
        finally:
            for proc in procs:
                _kill_server(proc)

    best = _best_result(topology_results)
    experiment = MultiServerExperimentResult(
        plugin=plugin_name,
        model_id=entry.vanilla_kwargs.get("hf_model_id", entry.model_id),
        served_model_id=entry.model_id,
        gpu=gpu,
        seq_len=entry.seq_len,
        num_requests=num_requests,
        server_counts=server_counts,
        per_server_concurrency_cap=per_server_concurrency,
        warmup_per_server=warmup_per_server,
        dataset_label=entry.dataset_label,
        results=topology_results,
        best_server_count=best.server_count if best is not None else None,
        best_req_per_s=best.req_per_s if best is not None else None,
        vllm_version=vllm_version,
    )
    saved_path = experiment.save(output_dir)

    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    for result in topology_results:
        if result.status == "ok":
            print(
                f"  {result.server_count} server(s):"
                f" req/s={result.req_per_s:.2f}"
                f" p95={result.p95_ms:.2f}ms"
                f" counts={result.per_server_request_counts}"
            )
        else:
            print(f"  {result.server_count} server(s): FAILED ({result.error})")
    if best is not None:
        print(
            f"  Sweet spot: {best.server_count} server(s)"
            f" at {best.req_per_s:.2f} req/s"
            f" (p95 {best.p95_ms:.2f}ms)"
        )
    else:
        print("  Sweet spot: none (all topologies failed)")
    print(f"  Result saved to {saved_path}")

    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental GLiNER2 multi-server benchmark")
    parser.add_argument("--plugin", default="deberta_gliner2")
    parser.add_argument("--server-counts", default="1,2,4,6", type=_parse_csv_ints)
    parser.add_argument("--per-server-concurrency", type=int, default=DEFAULT_PER_SERVER_CONCURRENCY)
    parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    parser.add_argument("--warmup-per-server", type=int, default=DEFAULT_WARMUP_PER_SERVER)
    parser.add_argument("--port-start", type=int, default=DEFAULT_PORT_START)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_experiment(
        plugin_name=args.plugin,
        server_counts=args.server_counts,
        per_server_concurrency=max(1, args.per_server_concurrency),
        num_requests=max(1, args.num_requests),
        warmup_per_server=max(1, args.warmup_per_server),
        port_start=args.port_start,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
