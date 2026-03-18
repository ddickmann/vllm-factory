"""
Nemotron ColEmbed Throughput Benchmark

Measures req/s, latency (p50/p95/p99) for the Nemotron ColEmbed plugin.
Supports --seq-len to control input token length for realistic benchmarks.

Usage:
    python plugins/nemotron_colembed/benchmark.py \
        --model <model-id> \
        --num-requests 1000 \
        --concurrency 32 \
        --seq-len 128
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import aiohttp

_BASE_WORDS = (
    "What is the significance of transformer architecture in modern deep learning "
    "applications including attention mechanisms self supervised pretraining and "
    "transfer learning for natural language processing computer vision and "
    "multimodal understanding tasks in production environments"
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


async def send_request(session: aiohttp.ClientSession, url: str, model: str, text: str) -> float:
    payload = {"model": model, "input": text}
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
        await asyncio.gather(*[send_request(session, url, model, texts[i % len(texts)]) for i in range(warmup)])

        print(f"  Timed run ({num_requests} requests, concurrency={concurrency})...")
        sem = asyncio.Semaphore(concurrency)
        start = time.perf_counter()

        async def bounded(text: str) -> float:
            async with sem:
                return await send_request(session, url, model, text)

        latencies = await asyncio.gather(*[bounded(texts[i % len(texts)]) for i in range(num_requests)])
        elapsed = time.perf_counter() - start

    latencies = sorted(latencies)
    n = len(latencies)
    return {
        "seq_len": seq_len,
        "total_requests": num_requests,
        "concurrency": concurrency,
        "elapsed_s": round(elapsed, 2),
        "req_per_sec": round(num_requests / elapsed, 1),
        "p50_ms": round(latencies[n // 2], 1),
        "p95_ms": round(latencies[int(n * 0.95)], 1),
        "p99_ms": round(latencies[int(n * 0.99)], 1),
        "mean_ms": round(statistics.mean(latencies), 1),
    }


def main():
    p = argparse.ArgumentParser(description="Nemotron ColEmbed throughput benchmark")
    p.add_argument("--model", default="nvidia/NV-Retriever-v2-embed")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--num-requests", type=int, default=500)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=128,
                    help="Approximate input length in tokens (default: 128)")
    args = p.parse_args()

    print(f"\nNemotron ColEmbed Benchmark (seq_len={args.seq_len})\n  model: {args.model}\n  requests: {args.num_requests}\n  concurrency: {args.concurrency}\n")
    results = asyncio.run(run_benchmark(
        args.base_url, args.model, args.num_requests,
        args.concurrency, args.warmup, args.seq_len,
    ))
    print("\n" + "=" * 60)
    print("Nemotron ColEmbed Benchmark Results")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:.<30} {v}")


if __name__ == "__main__":
    main()
