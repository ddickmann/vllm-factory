#!/usr/bin/env python3
"""
Benchmark: ModernColBERT — vLLM vs PyLate (vanilla)

Measures req/s for both at 512 tokens with batch=1,8,32.
"""
import gc
import os
import time

import torch
from transformers import AutoTokenizer

MODEL = "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT"

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

DOC_PREFIX_TOKEN_ID = 50369  # [D]

N_WARMUP = 10
N_RUNS = 20


def bench_vanilla():
    """PyLate vanilla baseline."""
    from pylate import models

    print("Loading PyLate ColBERT...")
    model = models.ColBERT(model_name_or_path=MODEL, document_length=512)

    # Warmup
    for _ in range(N_WARMUP):
        model.encode([TEXT_512], is_query=False)

    results = {}
    for bs in [1, 8, 32]:
        batch = [TEXT_512] * bs
        for _ in range(N_WARMUP):
            model.encode(batch, is_query=False)
        lats = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model.encode(batch, is_query=False)
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

    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    print("Loading vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    llm = LLM(
        model=MODEL, runner="pooling", trust_remote_code=True,
        enforce_eager=False, gpu_memory_utilization=0.5,
        max_model_len=512, enable_prefix_caching=False,
        disable_log_stats=True,
    )

    # Build pre-tokenized doc input
    encoded = tokenizer(TEXT_512, add_special_tokens=True, truncation=True,
                        max_length=511, padding=False, return_tensors=None)
    doc_ids = [encoded["input_ids"][0], DOC_PREFIX_TOKEN_ID] + encoded["input_ids"][1:]
    doc_ids = doc_ids[:512]
    seq_len = len(doc_ids)
    print(f"  Sequence length: {seq_len}")

    vram = torch.cuda.max_memory_allocated() / (1024**3)

    results = {"seq_len": seq_len, "vram_gib": round(vram, 2)}
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
    print("  ModernColBERT Benchmark — vLLM vs PyLate")
    print("=" * 70)

    print("\n[1/2] Vanilla (PyLate)...")
    try:
        vanilla = bench_vanilla()
    except Exception as e:
        print(f"  PyLate not available: {e}")
        vanilla = None

    print("\n[2/2] vLLM Factory...")
    vllm_r = bench_vllm()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS — ModernColBERT ({MODEL})")
    print(f"{'=' * 70}")
    print(f"\n  {'Metric':<25} {'Vanilla (PyLate)':<20} {'vLLM Factory':<20} {'Speedup'}")
    print(f"  {'-'*75}")
    for bs in [1, 8, 32]:
        v_rps = vanilla[f"batch{bs}_rps"] if vanilla else "N/A"
        l_rps = vllm_r[f"batch{bs}_rps"]
        speedup = f"{l_rps / v_rps:.2f}x" if vanilla else "N/A"
        print(f"  batch={bs:<3} req/s         {str(v_rps):<20} {l_rps:<20} {speedup}")
    if vanilla:
        print(f"  {'single p50':<25} {vanilla['single_p50_ms']:.1f}ms{'':<14} {vllm_r['single_p50_ms']:.1f}ms")
    print(f"  {'VRAM':<25} {'—':<20} {vllm_r['vram_gib']} GiB")
