#!/usr/bin/env python3
"""
Multi-instance throughput benchmark for encoder models.

Measures throughput scaling when running 1, 2, or 4 vLLM instances on a
single GPU via ``vllm-factory-serve --num-instances N``.

Usage:
    python benchmarks/benchmark_multi_instance.py
    python benchmarks/benchmark_multi_instance.py --plugins deberta_gliner2 --instances 1,2,4
    python benchmarks/benchmark_multi_instance.py --num-requests 500 --warmup 32
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

DEFAULT_PLUGINS = ["deberta_gliner2", "mmbert_gliner", "mt5_gliner"]
DEFAULT_INSTANCES = [1, 2, 4]
DEFAULT_NUM_REQUESTS = 200
DEFAULT_WARMUP = 32
DEFAULT_MAX_BS = 32
DEFAULT_PORT = 8000
DEFAULT_PORT_START = 9100
GPU_COOLDOWN_TIMEOUT = 60

OUTPUT_DIR = REPO_ROOT / "bench" / "results_multi_instance"


@dataclass
class RunResult:
    plugin: str
    model_id: str
    hf_model_id: str
    num_instances: int
    max_bs: int
    num_requests: int
    concurrency: int
    status: str
    req_per_s: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    gpu_util_per_instance: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkResult:
    timestamp: str
    gpu: str
    vllm_version: str
    runs: list[RunResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"multi_instance_{ts}.json"
        path.write_text(self.to_json())
        return path


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


def _gpu_memory_used_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def _wait_gpu_cooldown(baseline_mb: int, timeout_s: int = GPU_COOLDOWN_TIMEOUT) -> None:
    """Wait until GPU memory drops close to baseline."""
    threshold = baseline_mb + 500
    start = time.time()
    while time.time() - start < timeout_s:
        used = _gpu_memory_used_mb()
        if used <= threshold:
            return
        time.sleep(3)
    print(f"  [warn] GPU memory still at {_gpu_memory_used_mb()} MB "
          f"(baseline {baseline_mb} MB) after {timeout_s}s cooldown")


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its children via process group."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(3)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    except Exception as exc:
        print(f"  [warn] kill_process_tree({pid}): {exc}")


def _kill_gpu_orphans() -> None:
    """Kill any lingering processes on the GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        for line in out.strip().split("\n"):
            pid_str = line.strip().split(",")[0].strip()
            if pid_str.isdigit():
                try:
                    os.kill(int(pid_str), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
    except Exception:
        pass


def _available_gpu_util(num_instances: int) -> float:
    """Compute a safe gpu_memory_utilization based on actual free memory."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        used, total = (int(x) for x in out.strip().split("\n")[0].split(","))
        free = total - used
        headroom = 0.90
        per_inst = (free * headroom) / (total * num_instances)
        return max(0.10, round(min(0.80, per_inst), 2))
    except Exception:
        from forge.multi_instance import _scale_gpu_memory
        return _scale_gpu_memory(num_instances)


def _start_server(
    entry, num_instances: int, max_bs: int, port: int, port_start: int,
) -> subprocess.Popen:
    """Launch vllm-factory-serve as a subprocess."""
    gpu_util = _available_gpu_util(num_instances)
    cmd = [
        sys.executable, "-m", "forge.serve_cli",
        entry.model_id,
        "--num-instances", str(num_instances),
        "--max-batch-size", str(max_bs),
        "--port", str(port),
        "--port-start", str(port_start),
        "--gpu-memory-utilization", str(gpu_util),
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--io-processor-plugin", entry.io_plugin,
    ]
    env = os.environ.copy()
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    log_dir = REPO_ROOT / "bench" / "results_multi_instance" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{entry.plugin_name}_{num_instances}inst_{ts}.log"
    log_file = open(log_path, "w")

    return subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        text=True, start_new_session=True, cwd=str(REPO_ROOT),
    )


def _wait_healthy(port: int, timeout_s: int = 300) -> bool:
    """Poll /health until it returns 200."""
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


async def _run_load(
    entry, url: str, num_requests: int, concurrency: int, warmup: int,
) -> dict:
    """Send requests and measure throughput/latency."""
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0
    dataset = entry.get_dataset()

    async with aiohttp.ClientSession() as session:
        async def send(idx: int, is_warmup: bool = False) -> float | None:
            nonlocal errors
            payload = dataset[idx % len(dataset)]
            body = {"model": entry.model_id, entry.payload_key: payload}
            if entry.request_task is not None:
                body["task"] = entry.request_task

            async with sem:
                t0 = time.perf_counter()
                try:
                    async with session.post(
                        f"{url}{entry.endpoint}", json=body
                    ) as resp:
                        await resp.read()
                        if resp.status != 200:
                            errors += 1
                            return None
                except Exception:
                    errors += 1
                    return None
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if not is_warmup:
                    latencies.append(elapsed_ms)
                return elapsed_ms

        await asyncio.gather(*(send(i, is_warmup=True) for i in range(warmup)))

        start = time.perf_counter()
        await asyncio.gather(*(send(i) for i in range(num_requests)))
        wall_s = time.perf_counter() - start

    latencies.sort()
    n = len(latencies)
    if n == 0:
        return {"req_per_s": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0,
                "mean_ms": 0, "errors": errors}

    return {
        "req_per_s": round(num_requests / wall_s, 2) if wall_s > 0 else 0,
        "p50_ms": round(latencies[n // 2], 2),
        "p95_ms": round(latencies[min(n - 1, int(n * 0.95))], 2),
        "p99_ms": round(latencies[min(n - 1, int(n * 0.99))], 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "errors": errors,
    }


def run_single_topology(
    entry,
    num_instances: int,
    max_bs: int,
    num_requests: int,
    warmup: int,
    port: int,
    port_start: int,
) -> RunResult:
    """Run one (plugin, num_instances) pair end-to-end."""
    concurrency = max_bs * num_instances
    gpu_util = _available_gpu_util(num_instances)
    hf_id = entry.vanilla_kwargs.get("hf_model_id", entry.model_id)

    print(f"\n  [{entry.plugin_name}] {num_instances} instance(s), "
          f"concurrency={concurrency}, gpu_util={gpu_util:.2f}")

    proc = _start_server(entry, num_instances, max_bs, port, port_start)
    try:
        healthy_timeout = 120 + num_instances * 40
        if not _wait_healthy(port, timeout_s=healthy_timeout):
            return RunResult(
                plugin=entry.plugin_name, model_id=entry.model_id,
                hf_model_id=hf_id, num_instances=num_instances,
                max_bs=max_bs, num_requests=num_requests,
                concurrency=concurrency, status="failed",
                gpu_util_per_instance=gpu_util,
                error="health check timeout",
            )

        print(f"    Server healthy. Running {num_requests} requests...")
        url = f"http://127.0.0.1:{port}"
        metrics = asyncio.run(
            _run_load(entry, url, num_requests, concurrency, warmup)
        )

        result = RunResult(
            plugin=entry.plugin_name, model_id=entry.model_id,
            hf_model_id=hf_id, num_instances=num_instances,
            max_bs=max_bs, num_requests=num_requests,
            concurrency=concurrency, status="ok",
            req_per_s=metrics["req_per_s"],
            p50_ms=metrics["p50_ms"], p95_ms=metrics["p95_ms"],
            p99_ms=metrics["p99_ms"], mean_ms=metrics["mean_ms"],
            gpu_util_per_instance=gpu_util,
        )
        print(f"    {result.req_per_s:.1f} req/s  p50={result.p50_ms:.0f}ms  "
              f"p95={result.p95_ms:.0f}ms  errors={metrics['errors']}")
        return result

    except Exception as exc:
        return RunResult(
            plugin=entry.plugin_name, model_id=entry.model_id,
            hf_model_id=hf_id, num_instances=num_instances,
            max_bs=max_bs, num_requests=num_requests,
            concurrency=concurrency, status="failed",
            gpu_util_per_instance=gpu_util, error=str(exc),
        )
    finally:
        _kill_process_tree(proc.pid)
        time.sleep(5)
        _kill_gpu_orphans()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-instance throughput benchmark")
    parser.add_argument(
        "--plugins", default=",".join(DEFAULT_PLUGINS),
        help=f"Comma-separated plugin names (default: {','.join(DEFAULT_PLUGINS)})")
    parser.add_argument(
        "--instances", default=",".join(str(i) for i in DEFAULT_INSTANCES),
        help="Comma-separated instance counts (default: 1,2,4)")
    parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--max-bs", type=int, default=DEFAULT_MAX_BS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--port-start", type=int, default=DEFAULT_PORT_START)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    plugins = [p.strip() for p in args.plugins.split(",")]
    instances = [int(x.strip()) for x in args.instances.split(",")]

    gpu = _detect_gpu()
    vllm_version = _detect_vllm_version()
    gpu_baseline_mb = _gpu_memory_used_mb()

    print("=" * 72)
    print("  Multi-Instance Throughput Benchmark")
    print("=" * 72)
    print(f"  GPU:            {gpu}")
    print(f"  vLLM:           {vllm_version}")
    print(f"  Plugins:        {plugins}")
    print(f"  Instances:      {instances}")
    print(f"  Requests:       {args.num_requests}")
    print(f"  Max BS/inst:    {args.max_bs}")
    print(f"  GPU baseline:   {gpu_baseline_mb} MB")
    print("=" * 72)

    all_runs: list[RunResult] = []

    for plugin_name in plugins:
        entry = get_entry(plugin_name)

        if entry.prep_fn is not None:
            print(f"\n[{plugin_name}] Preparing model...")
            entry.prep_fn()

        for num_inst in instances:
            _wait_gpu_cooldown(gpu_baseline_mb)

            result = run_single_topology(
                entry=entry,
                num_instances=num_inst,
                max_bs=args.max_bs,
                num_requests=args.num_requests,
                warmup=args.warmup,
                port=args.port,
                port_start=args.port_start,
            )
            all_runs.append(result)

    experiment = BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        gpu=gpu,
        vllm_version=vllm_version,
        runs=all_runs,
    )
    saved = experiment.save(args.output)

    print("\n" + "=" * 72)
    print("  Results Summary")
    print("=" * 72)
    print(f"  {'Plugin':<20} {'Inst':>4} {'req/s':>8} {'p50':>7} "
          f"{'p95':>7} {'Status':>8}")
    print("  " + "-" * 60)

    by_plugin: dict[str, dict[int, RunResult]] = {}
    for r in all_runs:
        by_plugin.setdefault(r.plugin, {})[r.num_instances] = r

    for plugin_name, inst_map in by_plugin.items():
        for num_inst in sorted(inst_map.keys()):
            r = inst_map[num_inst]
            if r.status == "ok":
                print(f"  {r.plugin:<20} {r.num_instances:>4}x "
                      f"{r.req_per_s:>7.1f} {r.p50_ms:>6.0f}ms "
                      f"{r.p95_ms:>6.0f}ms {r.status:>8}")
            else:
                print(f"  {r.plugin:<20} {r.num_instances:>4}x "
                      f"{'---':>7} {'---':>7} {'---':>7} {'FAILED':>8}")

        baseline = inst_map.get(1)
        best = max(
            (r for r in inst_map.values() if r.status == "ok"),
            key=lambda r: r.req_per_s, default=None,
        )
        if baseline and best and baseline.status == "ok" and best.num_instances > 1:
            gain = best.req_per_s / baseline.req_per_s
            print(f"  {'':20} Best: {best.num_instances}x "
                  f"-> {gain:.2f}x throughput")
        print()

    print(f"  Saved to: {saved}")
    print("=" * 72)


if __name__ == "__main__":
    main()
