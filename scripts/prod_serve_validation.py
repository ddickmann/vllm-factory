#!/usr/bin/env python3
"""Production Serve Validation — tests /health, /v1/pooling, /v1/embeddings endpoints.

Tests 3 representative plugins covering different model types:
  1. deberta_gliner  — DeBERTa encoder, NER extraction via /v1/pooling
  2. embeddinggemma  — Gemma decoder, standard embeddings via /v1/embeddings
  3. moderncolbert   — ModernBERT encoder, ColBERT multi-vector via /v1/pooling

For each plugin:
  - Cold start timing
  - /health endpoint
  - Correct endpoint with representative payload
  - Error handling (bad payload)
  - Concurrent requests (quick load test)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

import aiohttp
import requests

from forge.server import ModelServer

TESTS = [
    {
        "name": "deberta_gliner",
        "model": "nvidia/gliner-PII",
        "port": 8401,
        "gliner_plugin": "deberta_gliner",
        "endpoint": "/pooling",
        "payload": {
            "input": "John Smith works at NVIDIA in Santa Clara. Email: john@nvidia.com",
            "extra_kwargs": {"entities": ["person", "organization", "location", "email"]},
            "task": "embed",
        },
    },
    {
        "name": "embeddinggemma",
        "model": "unsloth/embeddinggemma-300m",
        "port": 8402,
        "endpoint": "/v1/embeddings",
        "payload": {
            "input": "task: search result | query: What is machine learning?",
        },
    },
    {
        "name": "moderncolbert",
        "model": "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT",
        "port": 8403,
        "endpoint": "/pooling",
        "payload": {
            "input": [50281, 50368, 2061, 389, 253, 2022, 5765, 265, 7312, 1818, 34, 50282,
                      50284, 50284, 50284, 50284, 50284, 50284, 50284, 50284],
        },
    },
]


def test_health(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def test_endpoint(base_url: str, endpoint: str, model: str, payload: dict, served_model: str | None = None) -> dict:
    full_payload = {"model": served_model or model, **payload}
    try:
        r = requests.post(f"{base_url}{endpoint}", json=full_payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            has_data = "data" in data and len(data["data"]) > 0
            return {"status": "PASS", "http_code": 200, "has_data": has_data}
        return {"status": "FAIL", "http_code": r.status_code, "body": r.text[:200]}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def test_error_handling(base_url: str, endpoint: str, model: str, served_model: str | None = None) -> dict:
    try:
        r = requests.post(f"{base_url}{endpoint}", json={"model": served_model or model}, timeout=10)
        return {"status": "PASS" if r.status_code >= 400 else "WARN", "http_code": r.status_code}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


async def test_concurrency(base_url: str, endpoint: str, model: str, payload: dict, n: int = 20, concurrency: int = 8, served_model: str | None = None) -> dict:
    full_payload = {"model": served_model or model, **payload}
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)

    async def single(session):
        async with sem:
            t0 = time.perf_counter()
            async with session.post(f"{base_url}{endpoint}", json=full_payload) as resp:
                status = resp.status
                await resp.read()
            return time.perf_counter() - t0, status

    async with aiohttp.ClientSession(connector=connector) as session:
        results = await asyncio.gather(*[single(session) for _ in range(n)])

    latencies = sorted([r[0] * 1000 for r in results])
    failures = sum(1 for r in results if r[1] != 200)
    return {
        "status": "PASS" if failures == 0 else "FAIL",
        "total": n,
        "failures": failures,
        "p50_ms": round(latencies[len(latencies) // 2], 1),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1),
    }


def run_test(test_cfg: dict) -> dict:
    name = test_cfg["name"]
    model = test_cfg["model"]
    port = test_cfg["port"]
    base_url = f"http://localhost:{port}"

    print(f"\n{'=' * 70}")
    print(f"  Testing: {name} ({model})")
    print(f"{'=' * 70}")

    results = {"name": name, "model": model}

    server = ModelServer(
        name=f"prod-validate-{name}",
        model=model,
        port=port,
        enforce_eager=True,
        startup_timeout=120,
        health_check_interval=1.0,
        trust_remote_code=True,
        gpu_memory_utilization=0.3,
        gliner_plugin=test_cfg.get("gliner_plugin"),
    )

    t0 = time.perf_counter()
    try:
        server.start()
    except Exception as e:
        results["cold_start"] = "FAIL"
        results["error"] = str(e)
        print(f"  Cold start: FAIL ({e})")
        return results

    cold_start_s = time.perf_counter() - t0
    results["cold_start_s"] = round(cold_start_s, 1)
    print(f"  Cold start: {cold_start_s:.1f}s")

    served_model = server.model

    try:
        h = test_health(base_url)
        results["health"] = "PASS" if h else "FAIL"
        print(f"  /health:    {'PASS' if h else 'FAIL'}")

        ep = test_endpoint(base_url, test_cfg["endpoint"], model, test_cfg["payload"], served_model=served_model)
        results["endpoint"] = ep
        print(f"  Endpoint:   {ep['status']} (HTTP {ep.get('http_code', 'N/A')})")

        err = test_error_handling(base_url, test_cfg["endpoint"], model, served_model=served_model)
        results["error_handling"] = err
        print(f"  Errors:     {err['status']} (HTTP {err.get('http_code', 'N/A')})")

        conc = asyncio.run(test_concurrency(
            base_url, test_cfg["endpoint"], model, test_cfg["payload"],
            served_model=served_model,
        ))
        results["concurrency"] = conc
        print(f"  Concurrency: {conc['status']} (p50={conc['p50_ms']}ms, p99={conc['p99_ms']}ms, failures={conc['failures']}/{conc['total']})")

    finally:
        server.stop()

    all_pass = (
        results.get("health") == "PASS"
        and results.get("endpoint", {}).get("status") == "PASS"
        and results.get("concurrency", {}).get("status") == "PASS"
    )
    results["overall"] = "PASS" if all_pass else "FAIL"
    return results


def main():
    print("=" * 70)
    print("  Production Serve Validation")
    print("=" * 70)

    all_results = []
    for test_cfg in TESTS:
        result = run_test(test_cfg)
        all_results.append(result)

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for r in all_results:
        status = r.get("overall", "FAIL")
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  {icon}  {r['name']:30s}  cold_start={r.get('cold_start_s', 'N/A')}s")

    report_path = "/tmp/prod_serve_validation.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Report: {report_path}")

    failed = [r for r in all_results if r.get("overall") != "PASS"]
    if failed:
        print(f"\n  {len(failed)} plugin(s) FAILED serve validation")
        sys.exit(1)
    else:
        print(f"\n  All {len(all_results)} plugins PASSED serve validation")
        sys.exit(0)


if __name__ == "__main__":
    main()
