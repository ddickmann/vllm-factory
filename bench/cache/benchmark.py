"""
vLLM-first benchmark for GLiNER2 /pooling.

This benchmark only targets the new GLiNER2 vLLM contract:
  POST /pooling

It runs a scenario matrix across:
  - schema mode (entities/classifications/relations/structures/mixed)
  - text mode (short/varied/long)
  - concurrency levels
  - include_confidence and include_spans toggles

Metrics:
  - latency (p50/p90/p95/p99/min/max)
  - throughput (req/s, chars/s, estimated tokens/s)
  - status-code distribution and validation failures
  - average response payload bytes

Example:
  uv run --with tiktoken python benchmark-better.py --base-url <service-url> --model fastino/gliner2-large-v1 --requests 20 --warmup 2 --concurrency 1,8,16,32,64 --schema-modes entities,classifications,relations,structures,mixed --text-modes short,varied,long --include-confidence true --include-spans false --path /pooling --probe --gpu-hourly-price 0.80 --json benchmark_modal_vllm_l4_smoke_patched_original_final_2026-04-12.json
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx

try:
    import tiktoken
except ImportError:  # pragma: no cover - exercised via runtime setup
    tiktoken = None


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "fastino/gliner2-large-v1"
DEFAULT_TOKEN_ENCODING = "o200k_base"


ENTITIES_SCHEMA = {
    "entities": {
        "person": "Person names",
        "company": "Company names",
        "product": "Product names",
        "location": "Places and locations",
    }
}
CLASSIFICATIONS_SCHEMA = {
    "classifications": [
        {
            "task": "sentiment",
            "labels": ["positive", "negative", "neutral"],
        }
    ]
}
RELATIONS_SCHEMA = {
    "relations": {
        "works_for": "Employment relationship",
        "located_in": "Location relationship",
    }
}
STRUCTURES_SCHEMA = {
    "structures": {
        "product_summary": {
            "fields": [
                {"name": "name", "dtype": "str", "description": "Product name"},
                {"name": "launch_site", "dtype": "str", "description": "Launch location"},
                {"name": "highlights", "dtype": "list", "description": "Key points"},
            ]
        }
    }
}
MIXED_SCHEMA = {
    **ENTITIES_SCHEMA,
    **CLASSIFICATIONS_SCHEMA,
    **RELATIONS_SCHEMA,
    **STRUCTURES_SCHEMA,
}


TEXT_LIBRARY = [
    "Apple released iPhone 15 in Cupertino and analysts called the launch successful.",
    "John Smith works at NVIDIA in Santa Clara and reports to Jensen Huang.",
    "The support ticket is urgent and billing related due to a duplicated invoice.",
    "Tim Cook announced a product roadmap at Apple Park during the keynote.",
]


@dataclass(frozen=True)
class Sample:
    scenario: str
    request_index: int
    latency_ms: float
    text_len_chars: int
    est_tokens: int
    text_tokens_exact: int
    request_tokens_exact: int
    status_code: int
    ok: bool
    error: Optional[str] = None
    response_bytes: int = 0
    response_has_data: bool = False
    validate_error: Optional[str] = None


@dataclass(frozen=True)
class RoundSummary:
    scenario: str
    schema_mode: str
    text_mode: str
    include_confidence: bool
    include_spans: bool
    concurrency: int
    requests: int
    wall_time_s: float
    samples: Sequence[Sample]


def parse_int_list(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must not be empty")
    try:
        parsed = [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be integers") from exc
    if any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("values must be >= 1")
    return parsed


def parse_str_list(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("value must not be empty")
    return items


def pctl(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def make_text(seed: int, length: int) -> str:
    base = " ".join(TEXT_LIBRARY)
    if length <= len(base):
        return base[:length].rstrip()

    chunks = [base]
    while len("".join(chunks)) < length:
        chunks.append(f" run-{seed} ")
        chunks.append(base)
    return "".join(chunks)[:length].rstrip()


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def require_token_encoder(encoding_name: str):
    if tiktoken is None:
        raise SystemExit(
            "tiktoken is required for exact token accounting. "
            "Run with: uv run --with tiktoken python benchmark.py ..."
        )
    return tiktoken.get_encoding(encoding_name)


def count_exact_tokens(encoder, text: str) -> int:
    return len(encoder.encode(text, disallowed_special=()))


def canonicalize_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cost_per_million_requests(hourly_price: float, requests_per_sec: float) -> float:
    if requests_per_sec <= 0:
        return 0.0
    return hourly_price / (requests_per_sec * 3600.0) * 1_000_000.0


def cost_per_million_tokens(hourly_price: float, tokens_per_sec: float) -> float:
    if tokens_per_sec <= 0:
        return 0.0
    return hourly_price / ((tokens_per_sec * 3600.0) / 1_000_000.0)


def cost_for_duration_seconds(hourly_price: float, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return hourly_price * (duration_s / 3600.0)


def build_schema(schema_mode: str) -> Dict[str, Any]:
    if schema_mode == "entities":
        return ENTITIES_SCHEMA
    if schema_mode == "classifications":
        return CLASSIFICATIONS_SCHEMA
    if schema_mode == "relations":
        return RELATIONS_SCHEMA
    if schema_mode == "structures":
        return STRUCTURES_SCHEMA
    if schema_mode == "mixed":
        return MIXED_SCHEMA
    raise ValueError(f"Unsupported schema mode: {schema_mode}")


def expected_keys_for_schema(schema_mode: str) -> List[str]:
    if schema_mode == "entities":
        return ["entities"]
    if schema_mode == "classifications":
        return ["sentiment"]
    if schema_mode == "relations":
        return ["relation_extraction"]
    if schema_mode == "structures":
        return ["product_summary"]
    if schema_mode == "mixed":
        return ["entities", "sentiment", "relation_extraction", "product_summary"]
    return []


def _validate_entity_items(
    entities: Any,
    *,
    include_confidence: bool,
    include_spans: bool,
) -> Optional[str]:
    if not isinstance(entities, dict):
        return "entities payload is not an object"
    for label, items in entities.items():
        if not isinstance(items, list):
            return f"entity label '{label}' is not a list"
        for item in items:
            if include_confidence or include_spans:
                if not isinstance(item, dict):
                    return f"entity item for '{label}' should be an object"
                if "text" not in item:
                    return f"entity item for '{label}' is missing text"
                if include_confidence and "confidence" not in item:
                    return f"entity item for '{label}' is missing confidence"
                if include_spans and ("start" not in item or "end" not in item):
                    return f"entity item for '{label}' is missing span offsets"
            elif not isinstance(item, str):
                return f"entity item for '{label}' should be a string"
    return None


def _validate_classification_item(
    value: Any,
    *,
    include_confidence: bool,
    threshold: float | None = None,
) -> Optional[str]:
    if value is None:
        # Single-label classification may legitimately abstain when the best
        # score falls below the configured threshold.
        return None
    if include_confidence:
        if not isinstance(value, dict):
            return "classification result should be an object"
        if "label" not in value or "confidence" not in value:
            return "classification result is missing label/confidence"
        return None
    if not isinstance(value, str):
        return "classification result should be a string"
    return None


def _validate_relation_items(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return "relation_extraction payload is not an object"
    for relation_name, items in value.items():
        if not isinstance(items, list):
            return f"relation '{relation_name}' is not a list"
    return None


def _validate_structure_items(value: Any) -> Optional[str]:
    if not isinstance(value, list):
        return "structure result should be a list"
    for item in value:
        if not isinstance(item, dict):
            return "structure instance should be an object"
    return None


def build_text(text_mode: str, seed: int) -> str:
    if text_mode == "short":
        return TEXT_LIBRARY[seed % len(TEXT_LIBRARY)]
    if text_mode == "varied":
        return make_text(seed, 80 + ((seed % 6) * 120))
    if text_mode == "long":
        return make_text(seed, 1400)
    raise ValueError(f"Unsupported text mode: {text_mode}")


def build_payload(
    *,
    model: str,
    schema_mode: str,
    text_mode: str,
    include_confidence: bool,
    include_spans: bool,
    threshold: float,
    seed: int,
) -> Dict[str, Any]:
    return {
        "model": model,
        "data": {
            "text": build_text(text_mode, seed),
            "schema": build_schema(schema_mode),
            "threshold": threshold,
            "include_confidence": include_confidence,
            "include_spans": include_spans,
        },
    }


def validate_response(
    schema_mode: str,
    body: Dict[str, Any],
    *,
    include_confidence: bool,
    include_spans: bool,
    threshold: float | None = None,
) -> Optional[str]:
    payload = body.get("data") if isinstance(body.get("data"), dict) else body
    if not isinstance(payload, dict):
        return "response payload is not an object"
    for key in expected_keys_for_schema(schema_mode):
        if key not in payload:
            return f"missing key: {key}"
    if schema_mode in {"entities", "mixed"}:
        error = _validate_entity_items(
            payload.get("entities", {}),
            include_confidence=include_confidence,
            include_spans=include_spans,
        )
        if error:
            return error
    if schema_mode in {"classifications", "mixed"}:
        error = _validate_classification_item(
            payload.get("sentiment"),
            include_confidence=include_confidence,
            threshold=threshold,
        )
        if error:
            return error
    if schema_mode in {"relations", "mixed"}:
        error = _validate_relation_items(payload.get("relation_extraction"))
        if error:
            return error
    if schema_mode in {"structures", "mixed"}:
        error = _validate_structure_items(payload.get("product_summary"))
        if error:
            return error
    return None


async def time_request(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    encoder,
    *,
    scenario: str,
    request_index: int,
    schema_mode: str,
    include_confidence: bool,
    include_spans: bool,
) -> Sample:
    started = time.perf_counter()
    response: Optional[httpx.Response] = None
    text = payload["data"]["text"]
    payload_text = canonicalize_payload(payload)
    text_len = len(text)
    text_tokens_exact = count_exact_tokens(encoder, text)
    request_tokens_exact = count_exact_tokens(encoder, payload_text)
    try:
        response = await client.post(url, json=payload)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        body = response.json()
        validation_error = validate_response(
            schema_mode,
            body,
            include_confidence=include_confidence,
            include_spans=include_spans,
            threshold=payload["data"].get("threshold"),
        )
        valid = validation_error is None
        return Sample(
            scenario=scenario,
            request_index=request_index,
            latency_ms=elapsed_ms,
            text_len_chars=text_len,
            est_tokens=estimate_tokens(text),
            text_tokens_exact=text_tokens_exact,
            request_tokens_exact=request_tokens_exact,
            status_code=response.status_code,
            ok=valid,
            error=None if valid else validation_error,
            response_bytes=len(response.content),
            response_has_data=isinstance(body.get("data"), dict) or isinstance(body, dict),
            validate_error=validation_error,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        status_code = response.status_code if response is not None else 0
        response_bytes = len(response.content) if response is not None else 0
        return Sample(
            scenario=scenario,
            request_index=request_index,
            latency_ms=elapsed_ms,
            text_len_chars=text_len,
            est_tokens=estimate_tokens(text),
            text_tokens_exact=text_tokens_exact,
            request_tokens_exact=request_tokens_exact,
            status_code=status_code,
            ok=False,
            error=str(exc),
            response_bytes=response_bytes,
            response_has_data=False,
            validate_error=None,
        )


async def run_round(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    encoder,
    scenario: str,
    schema_mode: str,
    text_mode: str,
    include_confidence: bool,
    include_spans: bool,
    threshold: float,
    requests: int,
    concurrency: int,
) -> List[Sample]:
    sem = asyncio.Semaphore(concurrency)
    samples: List[Sample] = []

    async def worker(idx: int) -> None:
        payload = build_payload(
            model=model,
            schema_mode=schema_mode,
            text_mode=text_mode,
            include_confidence=include_confidence,
            include_spans=include_spans,
            threshold=threshold,
            seed=idx,
        )
        async with sem:
            samples.append(
                await time_request(
                    client,
                    url,
                    payload,
                    encoder,
                    scenario=scenario,
                    request_index=idx,
                    schema_mode=schema_mode,
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                )
            )

    await asyncio.gather(*(worker(i) for i in range(requests)))
    return samples


def summarize(
    samples: Sequence[Sample],
    wall_time_s: float,
    *,
    gpu_hourly_price: float | None = None,
) -> Dict[str, Any]:
    latencies = [sample.latency_ms for sample in samples]
    total_chars = sum(sample.text_len_chars for sample in samples)
    total_tokens = sum(sample.est_tokens for sample in samples)
    total_text_tokens_exact = sum(sample.text_tokens_exact for sample in samples)
    total_request_tokens_exact = sum(sample.request_tokens_exact for sample in samples)
    total_errors = sum(1 for sample in samples if not sample.ok)
    total_status_errors = sum(1 for sample in samples if sample.status_code != 200)
    statuses = collections.Counter(sample.status_code for sample in samples)
    if not samples:
        return {}

    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    good_samples = [sample for sample in samples if sample.ok]
    summary = {
        "count": len(samples),
        "ok": len(good_samples),
        "errors": total_errors,
        "status_errors": total_status_errors,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p90_ms": pctl(latencies, 0.90),
        "p95_ms": pctl(latencies, 0.95),
        "p99_ms": pctl(latencies, 0.99),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "requests_per_sec": len(samples) / wall_time_s if wall_time_s > 0 else 0.0,
        "chars_per_sec": total_chars / wall_time_s if wall_time_s > 0 else 0.0,
        "est_tokens_per_sec": total_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        "text_tokens_per_sec_exact": total_text_tokens_exact / wall_time_s if wall_time_s > 0 else 0.0,
        "request_tokens_per_sec_exact": total_request_tokens_exact / wall_time_s if wall_time_s > 0 else 0.0,
        "total_text_tokens_exact": total_text_tokens_exact,
        "total_request_tokens_exact": total_request_tokens_exact,
        "avg_chars_per_request": total_chars / len(samples),
        "avg_tokens_per_request": total_tokens / len(samples),
        "avg_text_tokens_per_request_exact": total_text_tokens_exact / len(samples),
        "avg_request_tokens_per_request_exact": total_request_tokens_exact / len(samples),
        "avg_response_bytes": statistics.mean(sample.response_bytes for sample in samples),
        "status_counts": dict(statuses),
    }
    if gpu_hourly_price is not None:
        summary["gpu_hourly_price"] = gpu_hourly_price
        summary["benchmark_cost_usd"] = cost_for_duration_seconds(gpu_hourly_price, wall_time_s)
        summary["cost_per_request_usd"] = (
            summary["benchmark_cost_usd"] / len(samples) if samples else 0.0
        )
        summary["cost_per_million_requests"] = cost_per_million_requests(
            gpu_hourly_price, summary["requests_per_sec"]
        )
        summary["cost_per_million_request_tokens_exact"] = cost_per_million_tokens(
            gpu_hourly_price, summary["request_tokens_per_sec_exact"]
        )
    return summary


def print_summary(scenario: str, summary: Dict[str, Any]) -> None:
    print(f"\nScenario: {scenario}")
    print(
        f"  results: ok={summary['ok']}/{summary['count']} "
        f"errors={summary['errors']} status_errors={summary['status_errors']}"
    )
    print(
        f"  latency: mean={summary['mean_ms']:.2f}ms "
        f"median={summary['median_ms']:.2f}ms "
        f"p90={summary['p90_ms']:.2f}ms "
        f"p95={summary['p95_ms']:.2f}ms "
        f"p99={summary['p99_ms']:.2f}ms"
    )
    print(
        f"  throughput: {summary['requests_per_sec']:.2f} req/s "
        f"{summary['chars_per_sec']:.2f} chars/s "
        f"{summary['est_tokens_per_sec']:.2f} est_tok/s"
    )
    print(
        f"  request shape: {summary['avg_chars_per_request']:.1f} chars/request "
        f"{summary['avg_tokens_per_request']:.1f} est_tok/request"
    )
    print(
        f"  exact tokens: {summary['text_tokens_per_sec_exact']:.2f} text_tok/s "
        f"{summary['request_tokens_per_sec_exact']:.2f} request_tok/s "
        f"avg_request={summary['avg_request_tokens_per_request_exact']:.1f}"
    )
    if "cost_per_million_requests" in summary:
        print(
            f"  cost: run=${summary['benchmark_cost_usd']:.6f} "
            f"${summary['cost_per_request_usd']:.8f}/req "
            f"${summary['cost_per_million_requests']:.2f}/M req "
            f"${summary['cost_per_million_request_tokens_exact']:.5f}/M request_tok"
        )
    print(f"  response size: avg={summary['avg_response_bytes']:.1f} bytes")
    print(f"  statuses: {summary['status_counts']}")


def print_overall(results: Sequence[RoundSummary]) -> None:
    if not results:
        return
    total_requests = sum(len(item.samples) for item in results)
    total_ok = sum(sum(1 for sample in item.samples if sample.ok) for item in results)
    total_errors = total_requests - total_ok
    all_samples = [sample for item in results for sample in item.samples]
    all_latencies = [sample.latency_ms for sample in all_samples]
    print("\n" + "=" * 72)
    print("Overall Summary")
    print("=" * 72)
    print(f"Scenarios: {len(results)}")
    print(f"Requests: {total_requests}")
    print(f"Successful: {total_ok}")
    print(f"Failed: {total_errors}")
    print(
        f"Latency: mean={statistics.mean(all_latencies):.2f}ms "
        f"median={statistics.median(all_latencies):.2f}ms "
        f"p95={pctl(all_latencies, 0.95):.2f}ms "
        f"p99={pctl(all_latencies, 0.99):.2f}ms"
    )


def write_json_report(
    path: str,
    args: argparse.Namespace,
    results: Sequence[RoundSummary],
    *,
    session_elapsed_s: float,
) -> None:
    scenario_payloads = []
    for item in results:
        summary = summarize(
            item.samples,
            wall_time_s=item.wall_time_s,
            gpu_hourly_price=args.gpu_hourly_price,
        )
        scenario_payloads.append(
            {
                "scenario": item.scenario,
                "schema_mode": item.schema_mode,
                "text_mode": item.text_mode,
                "include_confidence": item.include_confidence,
                "include_spans": item.include_spans,
                "requests": item.requests,
                "concurrency": item.concurrency,
                "wall_time_s": item.wall_time_s,
                "summary": summary,
                "samples": [
                    {
                        "scenario": sample.scenario,
                        "request_index": sample.request_index,
                        "latency_ms": sample.latency_ms,
                        "text_len_chars": sample.text_len_chars,
                        "est_tokens": sample.est_tokens,
                        "text_tokens_exact": sample.text_tokens_exact,
                        "request_tokens_exact": sample.request_tokens_exact,
                        "status_code": sample.status_code,
                        "ok": sample.ok,
                        "error": sample.error,
                        "response_bytes": sample.response_bytes,
                        "response_has_data": sample.response_has_data,
                        "validate_error": sample.validate_error,
                    }
                    for sample in item.samples
                ],
            }
        )

    overall_samples = [sample for item in results for sample in item.samples]
    payload = {
        "base_url": args.base_url,
        "path": args.path,
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency_levels,
        "warmup": args.warmup,
        "schema_modes": args.schema_modes,
        "text_modes": args.text_modes,
        "threshold": args.threshold,
        "include_confidence": args.include_confidence,
        "include_spans": args.include_spans,
        "token_encoding": args.token_encoding,
        "gpu_hourly_price": args.gpu_hourly_price,
        "benchmark_session_elapsed_s": session_elapsed_s,
        "benchmark_session_cost_usd": (
            cost_for_duration_seconds(args.gpu_hourly_price, session_elapsed_s)
            if args.gpu_hourly_price is not None
            else None
        ),
        "overall_summary": summarize(
            overall_samples,
            wall_time_s=sum(item.wall_time_s for item in results),
            gpu_hourly_price=args.gpu_hourly_price,
        ),
        "scenarios": scenario_payloads,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def run_benchmark(args: argparse.Namespace) -> None:
    session_started = time.perf_counter()
    encoder = require_token_encoder(args.token_encoding)
    max_concurrency = max(args.concurrency_levels)
    limits = httpx.Limits(max_keepalive_connections=max_concurrency, max_connections=max_concurrency)
    timeout = httpx.Timeout(args.timeout)
    base_url = args.base_url.rstrip("/")
    url = f"{base_url}{args.path}"

    include_confidence_values = [False, True] if args.include_confidence == "both" else [args.include_confidence == "true"]
    include_spans_values = [False, True] if args.include_spans == "both" else [args.include_spans == "true"]

    print("=" * 72)
    print("GLiNER2 /pooling benchmark")
    print("=" * 72)
    print(f"URL: {url}")
    print(f"Model: {args.model}")
    print(f"Requests: {args.requests}")
    print(f"Concurrency levels: {args.concurrency_levels}")
    print(f"Warmup: {args.warmup}")
    print(f"Schema modes: {args.schema_modes}")
    print(f"Text modes: {args.text_modes}")
    print(f"include_confidence: {args.include_confidence}")
    print(f"include_spans: {args.include_spans}")
    print(f"Threshold: {args.threshold}")
    print(f"Token encoding: {args.token_encoding}")
    if args.gpu_hourly_price is not None:
        print(f"GPU hourly price: ${args.gpu_hourly_price:.2f}/h")

    if args.probe:
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as probe_client:
            health = await probe_client.get(f"{base_url}/health")
            health.raise_for_status()
            print("Probe /health: OK")
            models = await probe_client.get(f"{base_url}/v1/models")
            if models.status_code == 200:
                print("Probe /v1/models: OK")
            else:
                print(f"Probe /v1/models: status={models.status_code} (continuing)")

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        overall: List[RoundSummary] = []
        for schema_mode in args.schema_modes:
            for text_mode in args.text_modes:
                for include_confidence in include_confidence_values:
                    for include_spans in include_spans_values:
                        for concurrency in args.concurrency_levels:
                            scenario = (
                                f"schema={schema_mode} text={text_mode} "
                                f"conf={include_confidence} spans={include_spans} "
                                f"concurrency={concurrency}"
                            )
                            for warmup_index in range(args.warmup):
                                warmup_payload = build_payload(
                                    model=args.model,
                                    schema_mode=schema_mode,
                                    text_mode=text_mode,
                                    include_confidence=include_confidence,
                                    include_spans=include_spans,
                                    threshold=args.threshold,
                                    seed=warmup_index,
                                )
                                warmup_sample = await time_request(
                                    client,
                                    url,
                                    warmup_payload,
                                    encoder,
                                    scenario=scenario,
                                    request_index=warmup_index,
                                    schema_mode=schema_mode,
                                    include_confidence=include_confidence,
                                    include_spans=include_spans,
                                )
                                if not warmup_sample.ok:
                                    raise RuntimeError(
                                        f"Warmup failed in '{scenario}': "
                                        f"{warmup_sample.error or warmup_sample.status_code}"
                                    )

                            started = time.perf_counter()
                            samples = await run_round(
                                client=client,
                                url=url,
                                model=args.model,
                                encoder=encoder,
                                scenario=scenario,
                                schema_mode=schema_mode,
                                text_mode=text_mode,
                                include_confidence=include_confidence,
                                include_spans=include_spans,
                                threshold=args.threshold,
                                requests=args.requests,
                                concurrency=concurrency,
                            )
                            wall_s = time.perf_counter() - started
                            summary = summarize(
                                samples,
                                wall_time_s=wall_s,
                                gpu_hourly_price=args.gpu_hourly_price,
                            )

                            print_summary(scenario, summary)
                            print(f"  wall time: {wall_s * 1000.0:.2f}ms")
                            overall.append(
                                RoundSummary(
                                    scenario=scenario,
                                    schema_mode=schema_mode,
                                    text_mode=text_mode,
                                    include_confidence=include_confidence,
                                    include_spans=include_spans,
                                    concurrency=concurrency,
                                    requests=args.requests,
                                    wall_time_s=wall_s,
                                    samples=samples,
                                )
                            )

        print_overall(overall)
        session_elapsed_s = time.perf_counter() - session_started
        print(f"Benchmark session time: {session_elapsed_s:.2f}s")
        if args.gpu_hourly_price is not None:
            print(
                f"Benchmark session cost: "
                f"${cost_for_duration_seconds(args.gpu_hourly_price, session_elapsed_s):.6f}"
            )
        if args.json_path:
            write_json_report(
                args.json_path,
                args,
                overall,
                session_elapsed_s=session_elapsed_s,
            )
            print(f"\nWrote JSON report to {args.json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the GLiNER2 /pooling endpoint")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Server base URL")
    parser.add_argument("--path", default="/pooling", help="Endpoint path, e.g. /pooling or /v1/pooling")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id used in request payload")
    parser.add_argument("--requests", type=int, default=100, help="Timed requests per scenario")
    parser.add_argument("--warmup", type=int, default=15, help="Warmup requests per scenario")
    parser.add_argument(
        "--concurrency",
        dest="concurrency_levels",
        type=parse_int_list,
        default=[1, 4, 8],
        help="Comma-separated concurrency levels",
    )
    parser.add_argument(
        "--schema-modes",
        type=parse_str_list,
        default=["entities", "mixed"],
        help="Comma-separated schema modes: entities,classifications,relations,structures,mixed",
    )
    parser.add_argument(
        "--text-modes",
        type=parse_str_list,
        default=["varied", "long"],
        help="Comma-separated text modes: short,varied,long",
    )
    parser.add_argument(
        "--include-confidence",
        choices=["false", "true", "both"],
        default="both",
        help="Whether to include confidence in requests",
    )
    parser.add_argument(
        "--include-spans",
        choices=["false", "true", "both"],
        default="false",
        help="Whether to include spans in requests",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Request threshold",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Write a machine-readable summary to this file",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--token-encoding",
        default=DEFAULT_TOKEN_ENCODING,
        help="tiktoken encoding name used for exact token accounting",
    )
    parser.add_argument(
        "--gpu-hourly-price",
        type=float,
        default=None,
        help="Optional GPU hourly price for cost-normalized metrics",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Probe /health and /v1/models before running scenarios",
    )
    args = parser.parse_args()

    allowed_schema = {"entities", "classifications", "relations", "structures", "mixed"}
    unknown_schema = [x for x in args.schema_modes if x not in allowed_schema]
    if unknown_schema:
        raise SystemExit(f"Unsupported --schema-modes values: {unknown_schema}")

    allowed_text = {"short", "varied", "long"}
    unknown_text = [x for x in args.text_modes if x not in allowed_text]
    if unknown_text:
        raise SystemExit(f"Unsupported --text-modes values: {unknown_text}")

    if not (0.0 <= args.threshold <= 1.0):
        raise SystemExit("--threshold must be in [0.0, 1.0]")

    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
