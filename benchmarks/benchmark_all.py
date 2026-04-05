#!/usr/bin/env python3
"""
Benchmark: EmbeddingGemma — vLLM vs SentenceTransformers (vanilla)

Measures req/s for both implementations at ~512 tokens.
"""
import gc
import time

import torch
from transformers import AutoTokenizer

# Repeat text to guarantee 512+ tokens
_BASE = (
    "The European Central Bank announced a significant policy shift on Wednesday, "
    "raising interest rates by 25 basis points to combat persistent inflationary "
    "pressures across the eurozone. ECB President Christine Lagarde stated that "
    "the decision was unanimous among the governing council members. The rate hike, "
    "which brings the main refinancing rate to 4.75 percent, marks the tenth "
    "consecutive increase since the tightening cycle began in July 2022. Economic "
    "analysts from major financial institutions including Goldman Sachs, JPMorgan "
    "Chase, and Deutsche Bank had widely anticipated the move, though some expressed "
    "concern about the potential impact on already-slowing economic growth in "
    "peripheral eurozone economies. The German economy contracted by 0.3 percent in "
    "the previous quarter, while France and Italy showed only marginal growth. Despite "
    "these challenges, headline inflation in the eurozone remained at 5.3 percent in "
    "August, well above the ECB's 2 percent target. Core inflation, which excludes "
    "volatile energy and food prices, stood at 5.3 percent, unchanged from July. "
    "Lagarde emphasized that the ECB would continue to follow a data-dependent approach "
    "and that future rate decisions would be made on a meeting-by-meeting basis. "
    "Financial markets reacted relatively calmly to the announcement, with the euro "
    "trading essentially flat against the US dollar at 1.0742. European government "
    "bond yields edged slightly higher, with the German ten-year Bund yield rising "
    "to 2.65 percent. Stock markets across Europe showed mixed reactions, with the "
    "Euro Stoxx 50 index declining by 0.4 percent while the FTSE 100 in London "
    "gained 0.2 percent. Banking sector stocks generally outperformed, as higher "
    "rates typically boost net interest margins. The next ECB policy meeting is "
    "scheduled for October 26, and market pricing currently suggests a pause in "
    "the rate hiking cycle with only a 20 percent probability assigned to another "
    "increase. However several ECB officials have cautioned against prematurely "
    "declaring victory over inflation noting that wage growth and services inflation "
    "remain elevated across much of the currency bloc. Labor markets have shown "
    "surprising resilience with the eurozone unemployment rate at a record low of "
    "6.4 percent suggesting that underlying demand pressures may persist longer "
    "than originally anticipated by the central bank staff projections."
)
TEXT_512 = _BASE + " " + _BASE  # Double to ensure 512+ tokens

MODEL = "unsloth/embeddinggemma-300m"
N_WARMUP = 10
N_RUNS = 20


def bench_vanilla():
    """SentenceTransformers baseline."""
    from sentence_transformers import SentenceTransformer

    print("  Loading SentenceTransformer...")
    model = SentenceTransformer(MODEL, trust_remote_code=True)
    tokenizer = model.tokenizer
    tokens = tokenizer(TEXT_512, truncation=True, max_length=512, return_tensors="pt")
    seq_len = tokens["input_ids"].shape[1]
    print(f"  Sequence length: {seq_len}")

    # Warmup
    for _ in range(N_WARMUP):
        model.encode([TEXT_512])

    results = {"seq_len": seq_len}
    for bs in [1, 8, 32]:
        batch = [TEXT_512] * bs
        for _ in range(N_WARMUP):
            model.encode(batch)
        lats = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model.encode(batch)
            lats.append((time.perf_counter() - t0) * 1000)
        lats.sort()
        mean_lat = sum(lats) / len(lats)
        rps = bs / (mean_lat / 1000)
        results[f"batch{bs}_rps"] = round(rps, 1)
        results[f"batch{bs}_mean_ms"] = round(mean_lat, 1)
        if bs == 1:
            results["single_p50_ms"] = round(lats[len(lats) // 2], 1)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def bench_vllm():
    """vLLM benchmark."""
    from vllm import LLM

    print("  Loading vLLM...")
    llm = LLM(
        model=MODEL, trust_remote_code=True,
        dtype="float32", gpu_memory_utilization=0.5,
        enforce_eager=False, enable_prefix_caching=False,
        disable_log_stats=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokens = tokenizer(TEXT_512, truncation=True, max_length=512, return_tensors="pt")
    seq_len = tokens["input_ids"].shape[1]
    print(f"  Sequence length: {seq_len}")

    vram = torch.cuda.max_memory_allocated() / (1024**3)

    results = {"seq_len": seq_len, "vram_gib": round(vram, 2)}
    for bs in [1, 8, 32]:
        prompts = [TEXT_512] * bs
        for _ in range(N_WARMUP):
            llm.embed(prompts)
        lats = []
        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.embed(prompts)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
        lats.sort()
        mean_lat = sum(lats) / len(lats)
        rps = bs / (mean_lat / 1000)
        results[f"batch{bs}_rps"] = round(rps, 1)
        results[f"batch{bs}_mean_ms"] = round(mean_lat, 1)
        if bs == 1:
            results["single_p50_ms"] = round(lats[len(lats) // 2], 1)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("  EmbeddingGemma Benchmark — vLLM vs SentenceTransformers")
    print(f"  Model: {MODEL}")
    print("=" * 70)

    print("\n[1/2] Vanilla (SentenceTransformers)...")
    vanilla = bench_vanilla()

    print("\n[2/2] vLLM Factory...")
    vllm_r = bench_vllm()

    print(f"\n{'=' * 70}")
    print("  RESULTS — EmbeddingGemma")
    print(f"  Sequence length: {vanilla['seq_len']} tokens")
    print(f"{'=' * 70}")
    print(f"\n  {'Metric':<25} {'Vanilla (ST)':<20} {'vLLM Factory':<20} {'Speedup'}")
    print(f"  {'-'*75}")
    for bs in [1, 8, 32]:
        v_rps = vanilla[f"batch{bs}_rps"]
        l_rps = vllm_r[f"batch{bs}_rps"]
        speedup = l_rps / v_rps if v_rps > 0 else 0
        print(f"  batch={bs:<3} req/s         {v_rps:<20} {l_rps:<20} {speedup:.2f}x")
    print(f"  {'single p50':<25} {vanilla['single_p50_ms']:.1f}ms{'':<14} {vllm_r['single_p50_ms']:.1f}ms")
    print(f"  {'VRAM':<25} {'—':<20} {vllm_r['vram_gib']} GiB")
