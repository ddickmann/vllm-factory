"""ColQwen3 Throughput Benchmark

Measures req/s, p50, p95, and p99 end-to-end latency against a running
vLLM server. Run after starting the server with serve.py.

Usage:
    # 1. Start server (in another terminal):
    python plugins/colqwen3/serve.py --model <model-id>

    # 2. Run benchmark:
    python plugins/colqwen3/benchmark.py \\
        --model <model-id> \\
        --base-url http://localhost:8000 \\
        --num-requests 500 \\
        --concurrency 16

Inputs: configurable via --seq-len for token-length-controlled benchmarks.
"""

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
    """Send a single pooling request; return latency in milliseconds."""
    start = time.perf_counter()
    async with session.post(url, json={"model": model, "input": text}) as r:
        await r.json()
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

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup — critical for accurate results (JIT compilation, CUDA graphs, etc.)
        print(f"  Warming up ({warmup} requests, ~{seq_len} tokens/req)...")
        await asyncio.gather(*[
            send_request(session, url, model, texts[i % len(texts)])
            for i in range(warmup)
        ])

        # Timed run
        print(f"  Timed run ({num_requests} requests, concurrency={concurrency})...")
        sem = asyncio.Semaphore(concurrency)

        async def bounded_request(text: str) -> float:
            async with sem:
                return await send_request(session, url, model, text)

        t0 = time.perf_counter()
        latencies = await asyncio.gather(*[
            bounded_request(texts[i % len(texts)]) for i in range(num_requests)
        ])
        elapsed = time.perf_counter() - t0

    latencies = sorted(latencies)
    n = len(latencies)
    return {
        "req/s": round(num_requests / elapsed, 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(latencies[int(n * 0.95)], 1),
        "p99_ms": round(latencies[int(n * 0.99)], 1),
    }


def main():
    p = argparse.ArgumentParser(description="ColQwen3 throughput benchmark")
    p.add_argument("--model", required=True, help="Model ID (must match server)")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--num-requests", type=int, default=500)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=128,
                   help="Approximate input length in tokens (default: 128)")
    args = p.parse_args()

    print("\nColQwen3 Benchmark")
    print(f"  model:       {args.model}")
    print(f"  base_url:    {args.base_url}")
    print(f"  requests:    {args.num_requests}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  warmup:      {args.warmup}")
    print(f"  seq_len:     {args.seq_len}\n")

    results = asyncio.run(run_benchmark(
        args.base_url, args.model, args.num_requests,
        args.concurrency, args.warmup, args.seq_len,
    ))

    print()
    print("─" * 40)
    print(f"  req/s:   {results['req/s']}")
    print(f"  p50_ms:  {results['p50_ms']}")
    print(f"  p95_ms:  {results['p95_ms']}")
    print(f"  p99_ms:  {results['p99_ms']}")
    print("─" * 40)


if __name__ == "__main__":
    main()
