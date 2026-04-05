#!/usr/bin/env python3
"""
Benchmark: ColQwen3 — vLLM in-process (text-only, no vanilla reference)

Measures req/s for multimodal ColPali/ColQwen retrieval at ~512 tokens with batch=1,8,32.
"""
import gc
import time

import torch

TEXT_512 = (
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
    "rates typically boost net interest margins."
)

MODEL = "VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1"
N_WARMUP = 10
N_RUNS = 20


def bench_vllm():
    from transformers import AutoTokenizer
    from vllm import LLM

    print("  Loading vLLM...")
    llm = LLM(
        model=MODEL, runner="pooling", trust_remote_code=True,
        enforce_eager=False, dtype="bfloat16", gpu_memory_utilization=0.9,
        max_model_len=2048, enable_prefix_caching=False,
        disable_log_stats=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    encoded = tokenizer(TEXT_512, truncation=True, max_length=512, return_tensors=None)
    doc_ids = encoded["input_ids"][:512]

    vram = torch.cuda.max_memory_allocated() / (1024**3)

    results = {"seq_len": len(doc_ids), "vram_gib": round(vram, 2)}
    for bs in [1, 8, 32]:
        prompts = [{"prompt_token_ids": doc_ids}] * bs
        for _ in range(N_WARMUP):
            llm.encode(prompts, pooling_task="token_embed")
        lats = []
        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.encode(prompts, pooling_task="token_embed")
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
    print(f"  ColQwen3 Benchmark — vLLM ({MODEL})")
    print("=" * 70)

    r = bench_vllm()
    print(f"\n{'=' * 70}")
    print("  RESULTS — ColQwen3 (vLLM only, no vanilla reference)")
    print(f"{'=' * 70}")
    for bs in [1, 8, 32]:
        print(f"  batch={bs:<3} req/s: {r[f'batch{bs}_rps']}")
    print(f"  VRAM: {r['vram_gib']} GiB")
