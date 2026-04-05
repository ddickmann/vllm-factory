#!/usr/bin/env python3
"""
Benchmark: All GLiNER variants — vLLM vs vanilla GLiNER library

Measures req/s for all 3 GLiNER plugins + linker at ~512 tokens.
Runs vanilla GLiNER reference for each model too.
"""
import gc
import os
import re
import sys
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

LABELS = ["person", "organization", "location", "date", "monetary_value"]
THRESHOLD = 0.5
MAX_WIDTH = 12
ENT_TOKEN = "<<ENT>>"
SEP_TOKEN = "<<SEP>>"
WORD_PATTERN = re.compile(r'\w+(?:[-_]\w+)*|\S')

N_WARMUP = 10
N_RUNS = 20

MODELS = {
    "mmbert_gliner": {
        "hf_model": "VAGOsolutions/SauerkrautLM-GLiNER",
        "vllm_dir": "/tmp/sauerkraut-gliner-vllm",
    },
    "mt5_gliner": {
        "hf_model": "knowledgator/gliner-x-large",
        "vllm_dir": "/tmp/gliner-x-large-vllm",
    },
    "deberta_gliner": {
        "hf_model": "nvidia/gliner-PII",
        "vllm_dir": "/tmp/gliner-pii-vllm",
    },
}


def bench_vanilla_gliner(hf_model, text, labels, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Benchmark vanilla GLiNER library."""
    from gliner import GLiNER

    print(f"  Loading GLiNER({hf_model})...")
    model = GLiNER.from_pretrained(hf_model)
    model.eval()

    # Get token count for reference
    tokenizer = model.data_processor.transformer_tokenizer
    words = [m.group() for m in WORD_PATTERN.finditer(text)]
    prompt_list = [t for lab in labels for t in (ENT_TOKEN, lab)] + [SEP_TOKEN]
    input_words = prompt_list + words
    enc = tokenizer(input_words, is_split_into_words=True, return_tensors="pt",
                    truncation=True, padding=False)
    seq_len = enc["input_ids"].shape[1]

    # Warmup
    for _ in range(n_warmup):
        model.predict_entities(text, labels, threshold=THRESHOLD)

    results = {"seq_len": seq_len}
    for bs in [1, 8, 32]:
        lats = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            for _ in range(bs):
                model.predict_entities(text, labels, threshold=THRESHOLD)
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


def bench_vllm_gliner(plugin_name, model_dir, text, labels, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Benchmark vLLM GLiNER plugin."""
    try:
        __import__(f"plugins.{plugin_name}")
    except:
        pass

    from transformers import AutoTokenizer
    from vllm import LLM, PoolingParams
    from vllm.inputs import TokensPrompt

    print(f"  Loading vLLM({model_dir})...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Build GLiNER input
    words_with_pos = [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(text)]
    words = [w[0] for w in words_with_pos]
    text_length = len(words)

    prompt_list = [t for lab in labels for t in (ENT_TOKEN, lab)] + [SEP_TOKEN]
    prompt_len = len(prompt_list)
    input_words = prompt_list + words

    tokenized = tokenizer(input_words, is_split_into_words=True,
                          return_tensors='pt', truncation=True, padding=False)
    input_ids = tokenized['input_ids'][0]
    seq_len = len(input_ids)

    # Words mask
    word_ids_list = tokenized.word_ids(batch_index=0)
    word_ids = torch.tensor([w if w is not None else -1 for w in word_ids_list], dtype=torch.long)
    prev_word_ids = torch.roll(word_ids, 1, dims=0)
    prev_word_ids[0] = -1
    valid = (word_ids != -1) & (word_ids != prev_word_ids) & (word_ids >= prompt_len)
    words_mask = torch.zeros_like(word_ids)
    words_mask[valid] = word_ids[valid] - prompt_len + 1

    # Spans
    starts = torch.arange(text_length).unsqueeze(1)
    widths = torch.arange(MAX_WIDTH).unsqueeze(0)
    span_starts = starts.expand(-1, MAX_WIDTH)
    span_ends = span_starts + widths
    span_idx = torch.stack([span_starts, span_ends], dim=-1).view(-1, 2).unsqueeze(0)
    span_mask = ((span_starts < text_length) & (span_ends < text_length)).view(-1).unsqueeze(0)

    gliner_data = {
        "input_ids": input_ids.tolist(),
        "words_mask": words_mask.tolist(),
        "text_lengths": text_length,
        "span_idx": span_idx[0].tolist(),
        "span_mask": span_mask[0].tolist(),
    }
    # Add attention mask for mt5/deberta
    if "attention_mask" in tokenized:
        gliner_data["attention_mask"] = tokenized["attention_mask"][0].tolist()

    llm = LLM(
        model=model_dir, trust_remote_code=True, enforce_eager=False,
        dtype="bfloat16", enable_prefix_caching=False, disable_log_stats=True,
    )

    vram = torch.cuda.max_memory_allocated() / (1024**3)

    prompt = TokensPrompt(prompt_token_ids=input_ids.tolist())
    params = PoolingParams(extra_kwargs=gliner_data)

    results = {"seq_len": seq_len, "vram_gib": round(vram, 2)}
    for bs in [1, 8, 32]:
        prompts = [prompt] * bs
        for _ in range(n_warmup):
            llm.embed(prompts, pooling_params=params)
        lats = []
        for _ in range(n_runs):
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


def bench_linker_vllm(text, labels, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Benchmark vLLM linker plugin."""
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    from vllm.pooling_params import PoolingParams

    model_dir = "plugins/deberta_gliner_linker/_model_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    words = [m.group() for m in WORD_PATTERN.finditer(text)]
    tok_result = tokenizer(
        [words], is_split_into_words=True,
        return_tensors="pt", truncation=True, padding="longest",
    )
    input_ids = tok_result["input_ids"][0]
    seq_len = len(input_ids)

    word_ids_list = tok_result.word_ids(batch_index=0)
    words_mask = torch.zeros(len(word_ids_list), dtype=torch.long)
    prev_wid = -1
    for idx, wid in enumerate(word_ids_list):
        if wid is not None and wid != prev_wid:
            words_mask[idx] = wid + 1
            prev_wid = wid

    H = 1024
    label_embs = torch.randn(len(labels), H)

    extra = {
        "input_ids": input_ids.tolist(),
        "words_mask": words_mask.tolist(),
        "text_lengths": len(words),
        "labels_embeds": label_embs.tolist(),
    }

    llm = LLM(
        model=model_dir, trust_remote_code=True, dtype="float32",
        max_model_len=512, enforce_eager=False, enable_prefix_caching=False,
        gpu_memory_utilization=0.85, disable_log_stats=True,
    )

    vram = torch.cuda.max_memory_allocated() / (1024**3)

    prompt = TokensPrompt(prompt_token_ids=input_ids.tolist())
    params = PoolingParams(extra_kwargs=extra)

    results = {"seq_len": seq_len, "vram_gib": round(vram, 2)}
    for bs in [1, 8]:
        prompts = [prompt] * bs
        for _ in range(n_warmup):
            llm.embed(prompts, pooling_params=params)
        lats = []
        for _ in range(n_runs):
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


def run_single_model(plugin_name):
    """Run vanilla + vLLM benchmark for a single GLiNER model in subprocess."""
    cfg = MODELS[plugin_name]

    print(f"\n{'='*70}")
    print(f"  {plugin_name.upper()}")
    print(f"  HF: {cfg['hf_model']}")
    print(f"{'='*70}")

    # Vanilla
    print("\n  [Vanilla GLiNER]")
    vanilla = bench_vanilla_gliner(cfg["hf_model"], TEXT_512, LABELS)
    del_cuda()

    # vLLM
    print("\n  [vLLM Factory]")
    vllm_r = bench_vllm_gliner(plugin_name, cfg["vllm_dir"], TEXT_512, LABELS)

    return vanilla, vllm_r


def del_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "all"

    all_results = {}

    if target in ("all", "mmbert_gliner"):
        v, l = run_single_model("mmbert_gliner")
        all_results["mmbert_gliner"] = {"vanilla": v, "vllm": l}
        del_cuda()

    if target in ("all", "mt5_gliner"):
        v, l = run_single_model("mt5_gliner")
        all_results["mt5_gliner"] = {"vanilla": v, "vllm": l}
        del_cuda()

    if target in ("all", "deberta_gliner"):
        v, l = run_single_model("deberta_gliner")
        all_results["deberta_gliner"] = {"vanilla": v, "vllm": l}
        del_cuda()

    if target in ("all", "deberta_gliner_linker"):
        print(f"\n{'='*70}")
        print("  DEBERTA_GLINER_LINKER")
        print(f"{'='*70}")
        print("\n  [vLLM Factory]")
        l = bench_linker_vllm(TEXT_512, LABELS)
        all_results["deberta_gliner_linker"] = {"vanilla": None, "vllm": l}
        del_cuda()

    # Summary
    print(f"\n\n{'='*90}")
    print("  SUMMARY — GLiNER Benchmarks (512 tokens, RTX 2000 Ada)")
    print(f"{'='*90}\n")
    print(f"  {'Plugin':<25} {'Batch':<7} {'Vanilla req/s':<16} {'vLLM req/s':<14} {'Speedup':<10} {'VRAM'}")
    print(f"  {'-'*85}")
    for name, data in all_results.items():
        v = data["vanilla"]
        l = data["vllm"]
        for bs in [1, 8, 32]:
            key = f"batch{bs}_rps"
            if key not in l:
                continue
            v_rps = v[key] if v and key in v else "—"
            l_rps = l[key]
            if v and key in v and v[key] > 0:
                speedup = f"{l_rps / v[key]:.1f}x"
            else:
                speedup = "—"
            vram = l.get("vram_gib", "?")
            print(f"  {name:<25} {bs:<7} {str(v_rps):<16} {l_rps:<14} {speedup:<10} {vram} GiB")
