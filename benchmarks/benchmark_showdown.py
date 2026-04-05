#!/usr/bin/env python3
"""
Three-Model Throughput Showdown
===============================
Compares vanilla PyTorch references vs vLLM Factory for 3 text models.

Usage:
    # Phase 1: with gliner + sentence_transformers installed
    python benchmark_showdown.py --vanilla

    # Phase 2: after reinstalling vllm + pip install -e .
    python benchmark_showdown.py --vllm

    # Phase 3: compare
    python benchmark_showdown.py --compare
"""
import argparse
import gc
import json
import os
import re
import sys
import time

import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

N_WARMUP = 5
N_REQUESTS = 124
N_IMAGES = 12          # fewer for the 4B vision model — still representative
N_WARMUP_IMAGES = 2

VANILLA_RESULTS = "/tmp/showdown_vanilla.json"
VLLM_RESULTS = "/tmp/showdown_vllm.json"

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

LABELS = ["person", "organization", "location", "date", "monetary_value"]
THRESHOLD = 0.5
MAX_WIDTH = 12
ENT_TOKEN = "<<ENT>>"
SEP_TOKEN = "<<SEP>>"
WORD_PATTERN = re.compile(r'\w+(?:[-_]\w+)*|\S')

IMAGE_URLS = [
    "https://developer.download.nvidia.com/images/isaac/nvidia-isaac-lab-1920x1080.jpg",
    "https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/asr-nemo-canary-featured.jpg",
    "https://blogs.nvidia.com/wp-content/uploads/2023/02/genome-sequencing-helix.jpg",
]


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ======================================================================
# VANILLA BENCHMARKS
# ======================================================================

def vanilla_mt5_gliner():
    from gliner import GLiNER

    print("\n[mt5_gliner] Loading GLiNER(knowledgator/gliner-x-large)...")
    torch.set_default_device("cuda")
    model = GLiNER.from_pretrained("knowledgator/gliner-x-large", map_location="cuda")
    model.eval()

    tokenizer = model.data_processor.transformer_tokenizer
    words = [m.group() for m in WORD_PATTERN.finditer(TEXT_512)]
    prompt_list = [t for lab in LABELS for t in (ENT_TOKEN, lab)] + [SEP_TOKEN]
    enc = tokenizer(prompt_list + words, is_split_into_words=True,
                    return_tensors="pt", truncation=True, padding=False)
    seq_len = enc["input_ids"].shape[1]
    print(f"  Sequence length: {seq_len}")

    # Also prepare vLLM model dir while GLiNER is loaded
    vllm_dir = "/tmp/gliner-x-large-vllm"
    if not os.path.isdir(vllm_dir):
        print("  Preparing vLLM model directory...")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from plugins.mt5_gliner.parity_test import prepare_local_model
        prepare_local_model(model)

    print(f"  Warmup ({N_WARMUP})...")
    for _ in range(N_WARMUP):
        model.predict_entities(TEXT_512, LABELS, threshold=THRESHOLD)

    print(f"  Timing {N_REQUESTS} sequential requests...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REQUESTS):
        model.predict_entities(TEXT_512, LABELS, threshold=THRESHOLD)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rps = N_REQUESTS / elapsed
    print(f"  => {rps:.1f} req/s ({elapsed:.2f}s)")

    del model
    torch.set_default_device("cpu")
    cleanup()
    return {"req_s": round(rps, 2), "elapsed_s": round(elapsed, 2), "seq_len": seq_len}


def vanilla_lfm2_colbert():
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    MODEL = "LiquidAI/LFM2-ColBERT-350M"
    print(f"\n[lfm2_colbert] Loading HF AutoModel({MODEL})...")
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    proj_path = hf_hub_download(repo_id=MODEL, filename="1_Dense/model.safetensors")
    with safetensors.torch.safe_open(proj_path, framework="pt", device="cpu") as f:
        proj_weight = f.get_tensor("linear.weight").to(torch.bfloat16).cuda()

    tokens = tokenizer(TEXT_512, return_tensors="pt", truncation=True, max_length=512, padding=False)
    tokens_gpu = {k: v.cuda() for k, v in tokens.items()}
    seq_len = tokens["input_ids"].shape[1]
    print(f"  Sequence length: {seq_len}")

    print(f"  Warmup ({N_WARMUP})...")
    for _ in range(N_WARMUP):
        with torch.no_grad():
            out = model(**tokens_gpu)
        _ = out.last_hidden_state.squeeze(0) @ proj_weight.T

    print(f"  Timing {N_REQUESTS} sequential requests...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REQUESTS):
        with torch.no_grad():
            out = model(**tokens_gpu)
        hidden = out.last_hidden_state.squeeze(0)
        projected = hidden @ proj_weight.T
        _ = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rps = N_REQUESTS / elapsed
    print(f"  => {rps:.1f} req/s ({elapsed:.2f}s)")

    del model, proj_weight
    cleanup()
    return {"req_s": round(rps, 2), "elapsed_s": round(elapsed, 2), "seq_len": seq_len}


def vanilla_embeddinggemma():
    from sentence_transformers import SentenceTransformer

    MODEL = "unsloth/embeddinggemma-300m"
    print(f"\n[embeddinggemma] Loading SentenceTransformer({MODEL})...")
    model = SentenceTransformer(MODEL, trust_remote_code=True, device="cuda")

    tokenizer = model.tokenizer
    tokens = tokenizer(TEXT_512, truncation=True, max_length=512, return_tensors="pt")
    seq_len = tokens["input_ids"].shape[1]
    print(f"  Sequence length: {seq_len}")

    batch = [TEXT_512] * N_REQUESTS

    print(f"  Warmup ({N_WARMUP})...")
    for _ in range(N_WARMUP):
        model.encode(batch)

    print(f"  Timing {N_REQUESTS} requests (SentenceTransformers batching)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.encode(batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rps = N_REQUESTS / elapsed
    print(f"  => {rps:.1f} req/s ({elapsed:.2f}s)")

    del model
    cleanup()
    return {"req_s": round(rps, 2), "elapsed_s": round(elapsed, 2), "seq_len": seq_len}


def vanilla_nemotron_colembed():
    from transformers import AutoModel
    from transformers.image_utils import load_image

    MODEL = "nvidia/nemotron-colembed-vl-4b-v2"
    print(f"\n[nemotron_colembed] Loading HF AutoModel({MODEL})...")
    model = AutoModel.from_pretrained(
        MODEL, device_map="cuda", trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval()

    print(f"  Downloading {len(IMAGE_URLS)} sample images...")
    base_images = [load_image(url) for url in IMAGE_URLS]
    for i, img in enumerate(base_images):
        print(f"    Image {i}: {img.size}")

    images = [base_images[i % len(base_images)] for i in range(N_IMAGES)]

    print(f"  Warmup ({N_WARMUP_IMAGES})...")
    for _ in range(N_WARMUP_IMAGES):
        model.forward_images(base_images, batch_size=len(base_images))

    print(f"  Timing {N_IMAGES} images (forward_images, batch_size=4)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.forward_images(images, batch_size=4)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rps = N_IMAGES / elapsed
    print(f"  => {rps:.1f} req/s ({elapsed:.2f}s)")

    del model
    cleanup()
    return {"req_s": round(rps, 2), "elapsed_s": round(elapsed, 2), "n_images": N_IMAGES}


# ======================================================================
# VLLM BENCHMARKS
# ======================================================================

def vllm_mt5_gliner():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        __import__("plugins.mt5_gliner")
    except Exception:
        pass

    from transformers import AutoTokenizer
    from vllm import LLM, PoolingParams
    from vllm.inputs import TokensPrompt

    model_dir = "/tmp/gliner-x-large-vllm"
    print(f"\n[mt5_gliner] Loading vLLM({model_dir})...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    words_with_pos = [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(TEXT_512)]
    words = [w[0] for w in words_with_pos]
    text_length = len(words)

    prompt_list = [t for lab in LABELS for t in (ENT_TOKEN, lab)] + [SEP_TOKEN]
    prompt_len = len(prompt_list)
    input_words = prompt_list + words

    tokenized = tokenizer(input_words, is_split_into_words=True,
                          return_tensors='pt', truncation=True, padding=False)
    input_ids = tokenized['input_ids'][0]
    seq_len = len(input_ids)
    print(f"  Sequence length: {seq_len}")

    word_ids_list = tokenized.word_ids(batch_index=0)
    word_ids = torch.tensor([w if w is not None else -1 for w in word_ids_list], dtype=torch.long)
    prev_word_ids = torch.roll(word_ids, 1, dims=0)
    prev_word_ids[0] = -1
    valid = (word_ids != -1) & (word_ids != prev_word_ids) & (word_ids >= prompt_len)
    words_mask = torch.zeros_like(word_ids)
    words_mask[valid] = word_ids[valid] - prompt_len + 1

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
    if "attention_mask" in tokenized:
        gliner_data["attention_mask"] = tokenized["attention_mask"][0].tolist()

    llm = LLM(
        model=model_dir, trust_remote_code=True, enforce_eager=False,
        dtype="bfloat16", enable_prefix_caching=False, disable_log_stats=True,
    )

    prompt = TokensPrompt(prompt_token_ids=input_ids.tolist())
    params = PoolingParams(extra_kwargs=gliner_data)
    prompts = [prompt] * N_REQUESTS

    print(f"  Warmup ({N_WARMUP})...")
    for _ in range(N_WARMUP):
        llm.embed(prompts, pooling_params=params)

    print(f"  Timing {N_REQUESTS} requests (single batch)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.embed(prompts, pooling_params=params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rps = N_REQUESTS / elapsed
    print(f"  => {rps:.1f} req/s ({elapsed:.2f}s)")

    del llm
    cleanup()
    return {"req_s": round(rps, 2), "elapsed_s": round(elapsed, 2), "seq_len": seq_len}


def vllm_lfm2_colbert():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers import AutoTokenizer
    from vllm import LLM

    MODEL = "LiquidAI/LFM2-ColBERT-350M"
    print(f"\n[lfm2_colbert] Loading vLLM({MODEL})...")
    llm = LLM(
        model=MODEL, runner="pooling", trust_remote_code=True,
        enforce_eager=False, dtype="bfloat16", gpu_memory_utilization=0.5,
        max_model_len=512, enable_prefix_caching=False, disable_log_stats=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    encoded = tokenizer(TEXT_512, truncation=True, max_length=512, return_tensors=None)
    doc_ids = encoded["input_ids"][:512]
    seq_len = len(doc_ids)
    print(f"  Sequence length: {seq_len}")

    prompts = [{"prompt_token_ids": doc_ids}] * N_REQUESTS

    print(f"  Warmup ({N_WARMUP})...")
    for _ in range(N_WARMUP):
        llm.encode(prompts, pooling_task="token_embed")

    print(f"  Timing {N_REQUESTS} requests (single batch)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.encode(prompts, pooling_task="token_embed")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rps = N_REQUESTS / elapsed
    print(f"  => {rps:.1f} req/s ({elapsed:.2f}s)")

    del llm
    cleanup()
    return {"req_s": round(rps, 2), "elapsed_s": round(elapsed, 2), "seq_len": seq_len}


def vllm_embeddinggemma():
    from transformers import AutoTokenizer
    from vllm import LLM

    MODEL = "unsloth/embeddinggemma-300m"
    print(f"\n[embeddinggemma] Loading vLLM({MODEL})...")
    llm = LLM(
        model=MODEL, trust_remote_code=True,
        dtype="float32", gpu_memory_utilization=0.5,
        enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokens = tokenizer(TEXT_512, truncation=True, max_length=512, return_tensors="pt")
    seq_len = tokens["input_ids"].shape[1]
    print(f"  Sequence length: {seq_len}")

    prompts = [TEXT_512] * N_REQUESTS

    print(f"  Warmup ({N_WARMUP})...")
    for _ in range(N_WARMUP):
        llm.embed(prompts)

    print(f"  Timing {N_REQUESTS} requests (single batch)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.embed(prompts)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rps = N_REQUESTS / elapsed
    print(f"  => {rps:.1f} req/s ({elapsed:.2f}s)")

    del llm
    cleanup()
    return {"req_s": round(rps, 2), "elapsed_s": round(elapsed, 2), "seq_len": seq_len}


def vllm_nemotron_colembed():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers import AutoProcessor
    from transformers.image_utils import load_image
    from vllm import LLM

    MODEL = "nvidia/nemotron-colembed-vl-4b-v2"
    print(f"\n[nemotron_colembed] Loading vLLM({MODEL})...")
    llm = LLM(
        model=MODEL, runner="pooling", trust_remote_code=True,
        enforce_eager=False, dtype="bfloat16", gpu_memory_utilization=0.9,
        max_model_len=4096, enable_prefix_caching=False, disable_log_stats=True,
        limit_mm_per_prompt={"image": 1},
    )

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    print(f"  Downloading {len(IMAGE_URLS)} sample images...")
    base_images = [load_image(url) for url in IMAGE_URLS]

    image_inputs = []
    for i in range(N_IMAGES):
        img = base_images[i % len(base_images)]
        if img.mode != "RGB":
            img = img.convert("RGB")
        passage_text = "passage: "
        message = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": passage_text},
        ]}]
        prompt = proc.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": img},
        })

    print(f"  Warmup ({N_WARMUP_IMAGES})...")
    warmup_inputs = image_inputs[:3]
    for _ in range(N_WARMUP_IMAGES):
        llm.encode(warmup_inputs, pooling_task="token_embed")

    print(f"  Timing {N_IMAGES} images (single batch)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.encode(image_inputs, pooling_task="token_embed")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rps = N_IMAGES / elapsed
    print(f"  => {rps:.1f} req/s ({elapsed:.2f}s)")

    del llm
    cleanup()
    return {"req_s": round(rps, 2), "elapsed_s": round(elapsed, 2), "n_images": N_IMAGES}


# ======================================================================
# PHASES
# ======================================================================

ALL_MODELS = ["mt5_gliner", "lfm2_colbert", "embeddinggemma"]

VANILLA_FNS = {
    "mt5_gliner": vanilla_mt5_gliner,
    "lfm2_colbert": vanilla_lfm2_colbert,
    "embeddinggemma": vanilla_embeddinggemma,
    "nemotron_colembed": vanilla_nemotron_colembed,
}

VLLM_FNS = {
    "mt5_gliner": vllm_mt5_gliner,
    "lfm2_colbert": vllm_lfm2_colbert,
    "embeddinggemma": vllm_embeddinggemma,
    "nemotron_colembed": vllm_nemotron_colembed,
}


def run_vanilla(only=None):
    targets = [only] if only else ALL_MODELS
    print("=" * 70)
    print("  PHASE 1: Vanilla Reference Benchmarks")
    print(f"  Models: {', '.join(targets)}")
    print("=" * 70)

    # Load existing results to merge into
    results = {}
    if os.path.exists(VANILLA_RESULTS):
        with open(VANILLA_RESULTS) as f:
            results = json.load(f)

    for name in targets:
        results[name] = VANILLA_FNS[name]()
        cleanup()

    with open(VANILLA_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {VANILLA_RESULTS}")
    print_single("Vanilla", results)


def run_vllm(only=None):
    targets = [only] if only else ALL_MODELS
    print("=" * 70)
    print("  PHASE 2: vLLM Factory Benchmarks")
    print(f"  Models: {', '.join(targets)}")
    print("=" * 70)

    results = {}
    if os.path.exists(VLLM_RESULTS):
        with open(VLLM_RESULTS) as f:
            results = json.load(f)

    for name in targets:
        results[name] = VLLM_FNS[name]()
        cleanup()

    with open(VLLM_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {VLLM_RESULTS}")
    print_single("vLLM", results)


def print_single(label, results):
    print(f"\n{'=' * 50}")
    print(f"  {label} Results")
    print(f"{'=' * 50}")
    for name, r in results.items():
        print(f"  {name:<22} {r['req_s']:>8.1f} req/s  ({r['elapsed_s']:.2f}s)")


def run_compare():
    if not os.path.exists(VANILLA_RESULTS):
        print(f"Missing {VANILLA_RESULTS} — run --vanilla first")
        sys.exit(1)
    if not os.path.exists(VLLM_RESULTS):
        print(f"Missing {VLLM_RESULTS} — run --vllm first")
        sys.exit(1)

    with open(VANILLA_RESULTS) as f:
        vanilla = json.load(f)
    with open(VLLM_RESULTS) as f:
        vllm_r = json.load(f)

    print()
    print("=" * 70)
    print("  THROUGHPUT SHOWDOWN — Vanilla vs vLLM Factory")
    print(f"  {N_REQUESTS} requests per model, {N_WARMUP} warmup")
    print("=" * 70)
    print()
    print(f"  {'Model':<22} {'Vanilla':>12} {'vLLM':>12} {'Speedup':>10}")
    print(f"  {'-' * 58}")

    for name in ["mt5_gliner", "lfm2_colbert", "embeddinggemma"]:
        v = vanilla.get(name, {})
        l = vllm_r.get(name, {})
        v_rps = v.get("req_s", 0)
        l_rps = l.get("req_s", 0)
        speedup = l_rps / v_rps if v_rps > 0 else 0
        print(f"  {name:<22} {v_rps:>9.1f} r/s {l_rps:>9.1f} r/s {speedup:>8.1f}x")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Three-Model Throughput Showdown")
    parser.add_argument("--vanilla", action="store_true", help="Run vanilla references")
    parser.add_argument("--vllm", action="store_true", help="Run vLLM benchmarks")
    parser.add_argument("--compare", action="store_true", help="Compare results")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (e.g. nemotron_colembed)")
    args = parser.parse_args()

    if not any([args.vanilla, args.vllm, args.compare]):
        parser.print_help()
        sys.exit(1)

    if args.vanilla:
        run_vanilla(only=args.model)
    if args.vllm:
        run_vllm(only=args.model)
    if args.compare:
        run_compare()
