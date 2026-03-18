"""
DeBERTa-v2-GLiNER Throughput Benchmark

Measures req/s, latency (p50/p95/p99) for the DeBERTa-v2-GLiNER plugin.
Supports --seq-len to control input token length for realistic benchmarks.

Usage:
    python plugins/deberta_gliner/benchmark.py \
        --model <gliner-model-id> \
        --num-requests 1000 \
        --concurrency 32 \
        --seq-len 128
"""

import argparse
import asyncio
import statistics
import time
from typing import List

import aiohttp

_BASE_WORDS = (
    "John Smith john.smith@example.com works at NVIDIA Corporation in Santa Clara "
    "California since 2015 his phone number is 555-0123 and his home address is "
    "742 Evergreen Terrace Springfield Illinois 62704 United States of America "
    "contact his assistant Jane Doe at jane.doe@nvidia.com for scheduling"
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
    entities: List[str],
) -> float:
    payload = {
        "model": model,
        "input": text,
        "extra_kwargs": {"entities": entities},
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
    entities = ["person", "email", "phone_number", "address", "organization"]
    texts = _generate_texts(seq_len)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"  Warming up ({warmup} requests, ~{seq_len} tokens/req)...")
        warmup_tasks = [
            send_request(session, url, model, texts[i % len(texts)], entities)
            for i in range(warmup)
        ]
        await asyncio.gather(*warmup_tasks)

        print(f"  Timed run ({num_requests} requests, concurrency={concurrency})...")
        start = time.perf_counter()
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded(text: str) -> float:
            async with semaphore:
                return await send_request(session, url, model, text, entities)

        latencies = await asyncio.gather(*[
            bounded(texts[i % len(texts)]) for i in range(num_requests)
        ])
        elapsed = time.perf_counter() - start

    latencies = sorted(latencies)
    return {
        "seq_len": seq_len,
        "total_requests": num_requests,
        "concurrency": concurrency,
        "elapsed_s": round(elapsed, 2),
        "req_per_sec": round(num_requests / elapsed, 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1),
    }


def main():
    parser = argparse.ArgumentParser(description="DeBERTa-v2-GLiNER throughput benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Approximate input length in tokens (default: 128)")
    args = parser.parse_args()

    print(f"\nDeBERTa-v2-GLiNER Benchmark (seq_len={args.seq_len})")
    results = asyncio.run(run_benchmark(
        args.base_url, args.model, args.num_requests,
        args.concurrency, args.warmup, args.seq_len,
    ))
    print("\n" + "=" * 60)
    print("DeBERTa-v2-GLiNER Benchmark Results")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:.<30} {v}")


if __name__ == "__main__":
    main()
