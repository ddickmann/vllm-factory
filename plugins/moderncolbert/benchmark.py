"""
ModernColBERT Throughput Benchmark

Measures req/s, p50/p95/p99 latency for the ModernColBERT vLLM server.
Uses pre-tokenized inputs ([Q]/[D] prefix at position 1) sent via /v1/pooling
to exactly match the server's expected input format.

Usage:
    # Start server first:
    python plugins/moderncolbert/serve.py --model VAGOsolutions/SauerkrautLM-Multi-ModernColBERT

    # Then benchmark:
    python plugins/moderncolbert/benchmark.py \
        --model VAGOsolutions/SauerkrautLM-Multi-ModernColBERT \
        --base-url http://127.0.0.1:8100 \
        --num-requests 500 \
        --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import List

import aiohttp

# Query/Document prefix token IDs (ModernBERT [Q]/[D] special tokens)
QUERY_PREFIX_TOKEN_ID = 50368   # [Q] with trailing space
DOC_PREFIX_TOKEN_ID = 50369     # [D] with trailing space

# Hardcoded fallback token IDs for benchmark (no tokenizer needed)
# Standard ModernBERT token IDs: BOS=50281, EOS=50282
BOS_ID = 50281
EOS_ID = 50282


def _make_query_ids(n_tokens: int = 18) -> list:
    """Build a realistic pre-tokenized query input (no transformers dependency)."""
    # [BOS, [Q], text_tokens..., EOS] — n_tokens total
    text_ids = list(range(1000, 1000 + n_tokens - 3))  # Stable fake token IDs
    return [BOS_ID, QUERY_PREFIX_TOKEN_ID] + text_ids + [EOS_ID]


def _make_doc_ids(n_tokens: int = 80) -> list:
    """Build a realistic pre-tokenized document input."""
    # [BOS, [D], text_tokens..., EOS] — n_tokens total
    text_ids = list(range(2000, 2000 + n_tokens - 3))
    return [BOS_ID, DOC_PREFIX_TOKEN_ID] + text_ids + [EOS_ID]


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    token_ids: list,
) -> float:
    """Send a single /v1/pooling request, return latency in ms."""
    payload = {
        "model": model,
        "input": token_ids,  # Pre-tokenized integer IDs
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
    """Run the full throughput benchmark with mixed query/document inputs."""
    url = f"{base_url}/v1/pooling"

    # Queries at 32 tokens (realistic query length); documents at seq_len
    query_ids = [_make_query_ids(32) for i in range(50)]
    doc_ids = [_make_doc_ids(seq_len) for i in range(50)]
    all_inputs = query_ids + doc_ids  # 100 unique inputs

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        print(f"  Warming up ({warmup} requests, ~{seq_len} tokens/doc)...")
        warmup_tasks = [
            send_request(session, url, model, all_inputs[i % len(all_inputs)])
            for i in range(warmup)
        ]
        await asyncio.gather(*warmup_tasks)

        # Timed run
        print(f"  Timed run ({num_requests} requests, concurrency={concurrency})...")
        semaphore = asyncio.Semaphore(concurrency)
        start = time.perf_counter()

        async def bounded(token_ids: list) -> float:
            async with semaphore:
                return await send_request(session, url, model, token_ids)

        tasks = [bounded(all_inputs[i % len(all_inputs)]) for i in range(num_requests)]
        latencies: List[float] = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    latencies.sort()
    n = len(latencies)
    return {
        "req_per_s": round(num_requests / elapsed, 1),
        "p50_ms": round(latencies[n // 2], 1),
        "p95_ms": round(latencies[int(n * 0.95)], 1),
        "p99_ms": round(latencies[int(n * 0.99)], 1),
        "mean_ms": round(statistics.mean(latencies), 1),
        "total_requests": num_requests,
        "elapsed_s": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="ModernColBERT throughput benchmark")
    parser.add_argument("--model", default="VAGOsolutions/SauerkrautLM-Multi-ModernColBERT")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--num-requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Document length in tokens (default: 128)")
    args = parser.parse_args()

    print()
    print("ModernColBERT Benchmark")
    print(f"  model:       {args.model}")
    print(f"  base_url:    {args.base_url}")
    print(f"  requests:    {args.num_requests}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  warmup:      {args.warmup}")
    print(f"  seq_len:     {args.seq_len}")
    print()

    results = asyncio.run(
        run_benchmark(
            args.base_url,
            args.model,
            args.num_requests,
            args.concurrency,
            args.warmup,
            args.seq_len,
        )
    )

    line = chr(9472) * 42
    print(line)
    print(f"  req/s:   {results['req_per_s']}")
    print(f"  p50_ms:  {results['p50_ms']}")
    print(f"  p95_ms:  {results['p95_ms']}")
    print(f"  p99_ms:  {results['p99_ms']}")
    print(line)


if __name__ == "__main__":
    main()
