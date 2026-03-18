#!/usr/bin/env python3
"""
Benchmark: vLLM vs GLiNER Library for entity linking.

Runs in separate processes to avoid GPU memory conflicts.
Tests both single-text and batch throughput scenarios.
"""

import json
import multiprocessing as mp
import re
import time

import torch

HF_MODEL = "knowledgator/gliner-linker-large-v1.0"


def _resolve_model_path() -> str:
    from plugins.deberta_gliner_linker import get_model_path
    return get_model_path()

TEST_TEXTS = [
    "Apple announced new products in California. Michael Jordan joined the team.",
    "Google expanded its operations in New York City and hired many engineers.",
    "Tesla's CEO Elon Musk visited Berlin to inspect the new Gigafactory.",
    "Amazon acquired Whole Foods and Jeff Bezos became the richest person.",
    "Microsoft released Windows 11 at their headquarters in Redmond, Washington.",
    "Barack Obama gave a speech at Harvard University about climate change.",
    "Netflix hired Reed Hastings as CEO and expanded to Japan and Korea.",
    "Samsung unveiled the Galaxy S24 at their event in Seoul, South Korea.",
]
TEST_LABELS = ["company", "person", "location"]


def benchmark_gliner():
    """Benchmark GLiNER library native inference."""
    from gliner import GLiNER

    print("=" * 60)
    print("GLiNER Library Benchmark")
    print("=" * 60)

    model = GLiNER.from_pretrained(HF_MODEL, map_location="cuda")
    model.eval()

    # Warmup
    _ = model.predict_entities(TEST_TEXTS[0], TEST_LABELS)

    # Single text
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        result = model.predict_entities(TEST_TEXTS[0], TEST_LABELS)
    single_lat = (time.perf_counter() - t0) / N * 1000
    print(f"\nSingle text latency: {single_lat:.1f}ms")
    print(f"  Entities: {result}")

    # Batch
    t0 = time.perf_counter()
    for _ in range(N):
        results = model.inference(TEST_TEXTS, TEST_LABELS, batch_size=8)
    batch_lat = (time.perf_counter() - t0) / N * 1000
    batch_tput = len(TEST_TEXTS) / (batch_lat / 1000)
    print(f"\nBatch ({len(TEST_TEXTS)} texts) latency: {batch_lat:.1f}ms")
    print(f"  Throughput: {batch_tput:.1f} texts/sec")
    print(f"  Per-text: {batch_lat / len(TEST_TEXTS):.1f}ms")

    # Show some entities
    for i, (text, ents) in enumerate(zip(TEST_TEXTS[:3], results[:3])):
        print(f"\n  [{i}] '{text[:50]}...'")
        for e in ents:
            print(f"      {e['text']} → {e['label']} ({e['score']:.3f})")

    return {"single_ms": single_lat, "batch_ms": batch_lat, "batch_size": len(TEST_TEXTS)}


def benchmark_vllm():
    """Benchmark vLLM pipeline."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, DebertaConfig, DebertaModel
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    from vllm.pooling_params import PoolingParams

    print("=" * 60)
    print("vLLM Pipeline Benchmark")
    print("=" * 60)

    # Precompute label embeddings (one-time cost, not measured)
    cfg_path = hf_hub_download(HF_MODEL, "gliner_config.json")
    with open(cfg_path) as f:
        gliner_cfg = json.load(f)
    le_cfg = gliner_cfg["labels_encoder_config"]
    labels_model = DebertaModel(DebertaConfig(**le_cfg))
    weights_path = hf_hub_download(HF_MODEL, "pytorch_model.bin")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    labels_prefix = "token_rep_layer.labels_encoder.model."
    labels_state = {k[len(labels_prefix):]: v for k, v in state.items() if k.startswith(labels_prefix)}
    labels_model.load_state_dict(labels_state, strict=False)
    labels_model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(_resolve_model_path(), use_fast=True)
    label_embs = []
    for label in TEST_LABELS:
        enc = tokenizer(label, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = labels_model(input_ids=enc["input_ids"].cuda())
        hs = out.last_hidden_state
        mask_exp = enc["attention_mask"].cuda().unsqueeze(-1).expand(hs.size()).float()
        mean = (hs * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        label_embs.append(mean.squeeze(0).cpu())
    label_embs = torch.stack(label_embs, dim=0)  # (C, H)
    print(f"Label embeddings precomputed: {label_embs.shape}")

    del labels_model, state
    torch.cuda.empty_cache()

    # Create vLLM engine
    llm = LLM(
        model=_resolve_model_path(),
        trust_remote_code=True,
        dtype="float32",
        max_model_len=512,
        enforce_eager=False,
        disable_log_stats=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
    )

    word_pattern = re.compile(r'\w+(?:[-_]\w+)*|\S')

    def prepare_input(text):
        words = [m.group() for m in word_pattern.finditer(text)]
        tok_result = tokenizer(
            [words], is_split_into_words=True,
            return_tensors="pt", truncation=True, padding="longest",
        )
        input_ids = tok_result["input_ids"][0]

        word_ids_list = tok_result.word_ids(batch_index=0)
        words_mask = torch.zeros(len(word_ids_list), dtype=torch.long)
        prev_wid = -1
        for idx, wid in enumerate(word_ids_list):
            if wid is not None and wid != prev_wid:
                words_mask[idx] = wid + 1
                prev_wid = wid

        gliner_data = {
            "input_ids": input_ids.tolist(),
            "words_mask": words_mask.tolist(),
            "text_lengths": len(words),
            "labels_embeds": label_embs.tolist(),
        }
        prompt = TokensPrompt(prompt_token_ids=input_ids.tolist())
        pooling_params = PoolingParams(extra_kwargs=gliner_data)
        return prompt, pooling_params

    # Prepare all inputs
    all_prompts = []
    all_params = []
    for text in TEST_TEXTS:
        p, pp = prepare_input(text)
        all_prompts.append(p)
        all_params.append(pp)

    # Warmup (20 iterations to ensure Triton JIT + CUDA graphs are fully cached)
    for _ in range(20):
        llm.embed([all_prompts[0]], pooling_params=all_params[0])

    # Single text
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        llm.embed([all_prompts[0]], pooling_params=all_params[0])
    single_lat = (time.perf_counter() - t0) / N * 1000
    print(f"\nSingle text latency: {single_lat:.1f}ms")

    # Batch — send all at once
    t0 = time.perf_counter()
    for _ in range(N):
        llm.embed(all_prompts, pooling_params=all_params)
    batch_lat = (time.perf_counter() - t0) / N * 1000
    batch_tput = len(TEST_TEXTS) / (batch_lat / 1000)
    print(f"\nBatch ({len(TEST_TEXTS)} texts) latency: {batch_lat:.1f}ms")
    print(f"  Throughput: {batch_tput:.1f} texts/sec")
    print(f"  Per-text: {batch_lat / len(TEST_TEXTS):.1f}ms")

    return {"single_ms": single_lat, "batch_ms": batch_lat, "batch_size": len(TEST_TEXTS)}


if __name__ == "__main__":
    # Run GLiNER first
    q1 = mp.Queue()
    def run_gliner(q):
        r = benchmark_gliner()
        q.put(r)
    p1 = mp.Process(target=run_gliner, args=(q1,))
    p1.start()
    p1.join()
    gliner_results = q1.get() if not q1.empty() else None

    print("\n\n")

    # Run vLLM
    q2 = mp.Queue()
    def run_vllm(q):
        r = benchmark_vllm()
        q.put(r)
    p2 = mp.Process(target=run_vllm, args=(q2,))
    p2.start()
    p2.join()
    vllm_results = q2.get() if not q2.empty() else None

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    if gliner_results and vllm_results:
        print(f"\n{'Metric':<30} {'GLiNER':>10} {'vLLM':>10} {'Speedup':>10}")
        print("-" * 62)

        gs, vs = gliner_results["single_ms"], vllm_results["single_ms"]
        print(f"{'Single text (ms)':<30} {gs:>10.1f} {vs:>10.1f} {gs/vs:>9.1f}x")

        gb, vb = gliner_results["batch_ms"], vllm_results["batch_ms"]
        print(f"{'Batch {0} texts (ms)'.format(gliner_results['batch_size']):<30} {gb:>10.1f} {vb:>10.1f} {gb/vb:>9.1f}x")

        gt = gliner_results["batch_size"] / (gb / 1000)
        vt = vllm_results["batch_size"] / (vb / 1000)
        print(f"{'Throughput (texts/sec)':<30} {gt:>10.1f} {vt:>10.1f} {vt/gt:>9.1f}x")
    else:
        print("Could not collect results from both benchmarks")
