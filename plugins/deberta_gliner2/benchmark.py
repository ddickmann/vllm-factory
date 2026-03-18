"""
GLiNER2 Throughput Benchmark

Measures req/s, latency (p50/p95/p99) for the GLiNER2 plugin.
Supports --seq-len to control input token length for realistic benchmarks.

Usage:
    python plugins/deberta_gliner2/benchmark.py \
        --model <gliner-model-id> \
        --num-requests 500 \
        --concurrency 32 \
        --seq-len 128
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import aiohttp

SCHEMA = {
    "employee": {
        "name": "text", "title": "text", "company": "text",
        "location": "text", "email": "text",
    }
}

_BASE_WORDS = (
    "Jane Doe is the VP of Engineering at TechCorp headquartered in New York City "
    "her email is jane.doe@techcorp.com she previously worked as a Senior Director "
    "at Google in Mountain View California and holds a PhD in Computer Science from "
    "Stanford University her team consists of 150 engineers across three offices"
).split()


def _generate_texts(target_tokens: int, n: int = 100) -> list:
    target_words = max(8, int(target_tokens * 0.75))
    texts = []
    for i in range(n):
        words = []
        while len(words) < target_words:
            words.extend(_BASE_WORDS)
        words = words[:target_words]
        words[0] = f"Sample{i}"
        texts.append(" ".join(words))
    return texts


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    text: str,
) -> float:
    payload = {
        "model": model,
        "input": text,
        "extra_kwargs": {"schema": SCHEMA},
    }
    start = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        await resp.json()
    return (time.perf_counter() - start) * 1000


async def run_benchmark(
    base_url: str,
    model: str,
    num_requests: int,
    concurrency: int,
    warmup: int = 200,
    seq_len: int = 128,
) -> dict:
    url = f"{base_url}/v1/pooling"
    texts = _generate_texts(seq_len)

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"  Warming up ({warmup} requests, ~{seq_len} tokens/req)...")
        warmup_tasks = [
            send_request(session, url, model, texts[i % len(texts)])
            for i in range(warmup)
        ]
        await asyncio.gather(*warmup_tasks)

        print(f"  Timed run ({num_requests} requests, concurrency={concurrency})...")
        semaphore = asyncio.Semaphore(concurrency)
        start = time.perf_counter()

        async def bounded(text: str) -> float:
            async with semaphore:
                return await send_request(session, url, model, text)

        latencies = await asyncio.gather(*[
            bounded(texts[i % len(texts)]) for i in range(num_requests)
        ])
        elapsed = time.perf_counter() - start

    latencies = sorted(latencies)
    n = len(latencies)
    return {
        "seq_len": seq_len,
        "total_requests": num_requests,
        "concurrency": concurrency,
        "elapsed_s": round(elapsed, 2),
        "req_per_sec": round(num_requests / elapsed, 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(latencies[int(n * 0.95)], 1),
        "p99_ms": round(latencies[int(n * 0.99)], 1),
        "mean_ms": round(statistics.mean(latencies), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="GLiNER2 throughput benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--num-requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Approximate input length in tokens (default: 128)")
    args = parser.parse_args()

    print(f"\nGLiNER2 Benchmark (seq_len={args.seq_len})")
    print(f"  model: {args.model}\n  requests: {args.num_requests}\n  concurrency: {args.concurrency}\n")
    results = asyncio.run(run_benchmark(
        args.base_url, args.model, args.num_requests,
        args.concurrency, args.warmup, args.seq_len,
    ))
    print("\n" + "=" * 60)
    print("GLiNER2 Benchmark Results")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:.<30} {v}")


if __name__ == "__main__":
    main()
