"""Structured benchmark result schema with JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SweepPoint:
    mode: str
    concurrency: int
    target_arrival_rps: float | None

    vllm_req_per_s: float
    vllm_p50_ms: float
    vllm_p95_ms: float
    vllm_p99_ms: float

    vanilla_req_per_s: float
    vanilla_p50_ms: float
    vanilla_p95_ms: float
    vanilla_p99_ms: float

    throughput_factor: float
    latency_factor: float


@dataclass
class BenchResult:
    plugin: str
    model_id: str
    served_model_id: str
    gpu: str
    seq_len: int
    num_requests: int
    concurrency_levels: list[int]
    modes: list[str]
    sweeps: list[SweepPoint]
    parity_metric: str
    parity_score: float

    dataset_label: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dtype: str = "bfloat16"
    vllm_version: str = ""
    compat_mode: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gpu_slug = _slugify(self.gpu)
        model_slug = _slugify(self.model_id)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{self.plugin}_{model_slug}_{gpu_slug}_{ts}.json"
        path.write_text(self.to_json())
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> BenchResult:
        data = json.loads(Path(path).read_text())
        if "sweeps" in data:
            data["sweeps"] = [SweepPoint(**point) for point in data["sweeps"]]
            data.setdefault("dataset_label", "")
            return cls(**data)

        return cls._from_legacy(data)

    @classmethod
    def load_dir(cls, results_dir: str | Path) -> list[BenchResult]:
        results_dir = Path(results_dir)
        results = []
        for p in sorted(results_dir.glob("*.json")):
            try:
                results.append(cls.from_json(p))
            except (json.JSONDecodeError, TypeError):
                continue
        return results

    def sweeps_for_mode(self, mode: str) -> list[SweepPoint]:
        return sorted(
            [s for s in self.sweeps if s.mode == mode],
            key=lambda s: s.concurrency,
        )

    def best_sweep(self, mode: str) -> SweepPoint | None:
        sweeps = self.sweeps_for_mode(mode)
        if not sweeps:
            return None
        return max(sweeps, key=lambda s: s.throughput_factor)

    @classmethod
    def _from_legacy(cls, data: dict[str, Any]) -> BenchResult:
        concurrency = int(data.get("concurrency", 1))
        sweep = SweepPoint(
            mode="saturate",
            concurrency=concurrency,
            target_arrival_rps=None,
            vllm_req_per_s=float(data["vllm_req_per_s"]),
            vllm_p50_ms=float(data["vllm_p50_ms"]),
            vllm_p95_ms=float(data["vllm_p95_ms"]),
            vllm_p99_ms=float(data["vllm_p99_ms"]),
            vanilla_req_per_s=float(data["vanilla_req_per_s"]),
            vanilla_p50_ms=float(data["vanilla_p50_ms"]),
            vanilla_p95_ms=float(data["vanilla_p95_ms"]),
            vanilla_p99_ms=float(data["vanilla_p99_ms"]),
            throughput_factor=float(data["throughput_factor"]),
            latency_factor=float(data["latency_factor"]),
        )
        return cls(
            plugin=data["plugin"],
            model_id=data["model_id"],
            served_model_id=data.get("served_model_id", data["model_id"]),
            gpu=data["gpu"],
            seq_len=int(data["seq_len"]),
            num_requests=int(data["num_requests"]),
            concurrency_levels=[concurrency],
            modes=["saturate"],
            sweeps=[sweep],
            parity_metric=data["parity_metric"],
            parity_score=float(data["parity_score"]),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            dtype=data.get("dtype", "bfloat16"),
        )


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
