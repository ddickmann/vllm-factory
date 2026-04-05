#!/usr/bin/env python3
"""
Benchmark: GLiNER2 (DeBERTa-v3) — vLLM in-process

Measures req/s for entity extraction at ~512 tokens with batch=1,8,32.
"""
import gc
import os
import time

import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

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

MODEL_DIR = "/tmp/gliner2-vllm"
N_WARMUP = 10
N_RUNS = 20


def bench_vllm():
    from transformers import AutoTokenizer
    from vllm import LLM, PoolingParams
    from vllm.inputs import TokensPrompt

    from plugins.deberta_gliner2.processor import (
        build_schema_for_entities,
        preprocess,
    )

    print("  Loading vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    schema = build_schema_for_entities(["person", "organization", "location", "date", "monetary_value"])
    prep = preprocess(tokenizer, TEXT_512, schema)

    llm = LLM(
        model=MODEL_DIR, trust_remote_code=True, enforce_eager=False,
        dtype="float16", enable_prefix_caching=False, disable_log_stats=True,
    )

    vram = torch.cuda.max_memory_allocated() / (1024**3)
    prompt = TokensPrompt(prompt_token_ids=prep["input_ids"])
    params = PoolingParams(extra_kwargs=prep)

    seq_len = len(prep["input_ids"])
    print(f"  Sequence length: {seq_len}")

    results = {"seq_len": seq_len, "vram_gib": round(vram, 2)}
    for bs in [1, 8, 32]:
        prompts = [prompt] * bs
        for _ in range(N_WARMUP):
            llm.embed(prompts, pooling_params=params)
        lats = []
        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.embed(prompts, pooling_params=params)
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
    print("  GLiNER2 Benchmark — vLLM (DeBERTa-v3)")
    print("=" * 70)

    r = bench_vllm()
    print(f"\n{'=' * 70}")
    print("  RESULTS — GLiNER2")
    print(f"{'=' * 70}")
    for bs in [1, 8, 32]:
        print(f"  batch={bs:<3} req/s: {r[f'batch{bs}_rps']}")
    print(f"  VRAM: {r['vram_gib']} GiB")
