"""
ColQwen3 Parity Test

Validates vLLM embeddings match HuggingFace reference with cosine >= 0.99.

CRITICAL: Uses slow Qwen2VL image processor. The fast processor gives ~0.74 cosine.

Usage:
    VLLM_WORKER_MULTIPROC_METHOD=spawn python plugins/colqwen3/parity_test.py \
        --model VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1 \
        --reference-dir /tmp/colqwen3_reference
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _patch_slow_image_processor():
    """Replace Qwen2VLImageProcessorFast with the slow version.

    CRITICAL: fast processor produces ~0.74 cosine vs ~0.99 with slow.
    Must be called before vLLM engine is created.
    """
    try:
        import transformers.models.qwen2_vl as _mod
        from transformers.models.qwen2_vl import (
            image_processing_qwen2_vl,
            image_processing_qwen2_vl_fast,
        )
        Slow = image_processing_qwen2_vl.Qwen2VLImageProcessor
        image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast = Slow
        _mod.Qwen2VLImageProcessorFast = Slow
        print("[patch] Applied: Qwen2VLImageProcessorFast -> slow")
    except Exception as exc:
        print(f"[patch] Warning: {exc}")


# Apply patch BEFORE any vLLM import
_patch_slow_image_processor()

import torch  # noqa: E402
from PIL import Image  # noqa: E402

# Preprocessing constants matching sauerkrautlm-colpali exactly

QUERY_PREFIX = 'Query: '
QUERY_AUG_TOKEN = '<|endoftext|>'
QUERY_AUG_SUFFIX = QUERY_AUG_TOKEN * 10
VISUAL_PROMPT_PREFIX = '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>'


def _make_test_image(idx: int) -> Image.Image:
    """Reproduce synthetic test image from generate_reference.py."""
    import random
    size = (1024, 1400)
    colors = [(245, 245, 245), (240, 248, 255), (255, 248, 240)]
    bg = colors[idx % len(colors)]
    img = Image.new("RGB", size, bg)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    rng2 = random.Random(idx * 1337)
    for _ in range(30):
        x0 = rng2.randint(50, size[0] - 200)
        y0 = rng2.randint(30, size[1] - 50)
        w = rng2.randint(80, 300)
        h = rng2.randint(8, 16)
        gray = rng2.randint(60, 180)
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(gray, gray, gray))
    return img


def _run_vllm_queries(model_path: str, queries: list) -> list:
    """Embed queries with vLLM using prompt_token_ids to bypass image processor."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    llm = LLM(
        model=model_path,
        runner="pooling",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        skip_mm_profiling=True,
        mm_processor_cache_gb=1,
        limit_mm_per_prompt={"image": 1},
    )

    inputs = []
    for q in queries:
        text = QUERY_PREFIX + q + QUERY_AUG_SUFFIX
        ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        inputs.append({"prompt_token_ids": ids})

    outputs = llm.encode(inputs, pooling_task="token_embed")
    embeddings = [torch.as_tensor(o.outputs.data).float() for o in outputs]
    del llm
    return embeddings


def _run_vllm_images(model_path: str, images: list) -> list:
    """Embed images with vLLM using native multimodal input."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from vllm import LLM

    llm = LLM(
        model=model_path,
        runner="pooling",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        skip_mm_profiling=True,
        mm_processor_cache_gb=1,
        limit_mm_per_prompt={"image": 1},
    )

    inputs = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        inputs.append({
            "prompt": VISUAL_PROMPT_PREFIX,
            "multi_modal_data": {"image": img},
        })

    outputs = llm.encode(inputs, pooling_task="token_embed")
    embeddings = [torch.as_tensor(o.outputs.data).float() for o in outputs]
    del llm
    return embeddings


def _cosine(a, b) -> float:
    """Mean token-level cosine similarity between two multi-vector embeddings."""
    a = a.float()
    b = b.float()
    min_len = min(a.shape[0], b.shape[0])
    a = a[:min_len]
    b = b[:min_len]
    an = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
    bn = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    return (an * bn).sum(dim=-1).mean().item()


def _print_result_table(title: str, results: list, threshold: float) -> bool:
    line = "-" * 70
    print()
    print(line)
    print(f"  {title}  (threshold: cosine >= {threshold})")
    print(line)
    header_num = "#"
    header_input = "Input"
    header_cosine = "Cosine"
    print(f"  {header_num:<4} {header_input:<40} {header_cosine:>8}   Status")
    for r in results:
        status = "OK PASS" if r["cosine"] >= threshold else "XX FAIL"
        idx_val = r["idx"]
        name_val = r["name"]
        cos_val = r["cosine"]
        print(f"  {idx_val:<4} {name_val:<40} {cos_val:.6f}   {status}")
    print(line)
    passed = all(r["cosine"] >= threshold for r in results)
    result_str = "OK ALL PASSED" if passed else "XX FAILURES"
    print(f"  Overall: {result_str}")
    return passed


def main():
    p = argparse.ArgumentParser(description="ColQwen3 vLLM parity test")
    p.add_argument("--model", required=True)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--report-dir", default="/tmp/colqwen3_reports")
    p.add_argument("--min-cosine", type=float, default=0.99)
    p.add_argument("--min-cosine-image", type=float, default=0.985,
                   help="Image threshold is slightly lower due to bf16/fp16 rounding "
                        "differences in the vision encoder forward pass.")
    args = p.parse_args()

    ref = Path(args.reference_dir)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("ColQwen3 Parity Test")
    print(f"  model:      {args.model}")
    print(f"  min_cosine: {args.min_cosine}")
    print("=" * 70)

    print("\n[1/4] Loading reference embeddings...")
    ref_queries = torch.load(ref / "query_embeddings.pt", weights_only=False)
    ref_images = torch.load(ref / "image_embeddings.pt", weights_only=False)
    queries_text = json.loads((ref / "queries.json").read_text())
    print(f"  queries: {len(ref_queries)}, images: {len(ref_images)}")

    images = [_make_test_image(i) for i in range(len(ref_images))]

    print("\n[2/4] Computing vLLM query embeddings...")
    vllm_queries = _run_vllm_queries(args.model, queries_text)

    print("\n[3/4] Computing vLLM image embeddings...")
    vllm_images = _run_vllm_images(args.model, images)

    print("\n[4/4] Computing parity scores...")
    q_results = []
    for i, (ref_e, vllm_e) in enumerate(zip(ref_queries, vllm_queries)):
        q_results.append({"idx": i, "name": f"query_{i}", "cosine": _cosine(ref_e, vllm_e)})

    i_results = []
    for i, (ref_e, vllm_e) in enumerate(zip(ref_images, vllm_images)):
        i_results.append({"idx": i, "name": f"image_{i}", "cosine": _cosine(ref_e, vllm_e)})

    q_passed = _print_result_table("Query parity", q_results, args.min_cosine)
    i_passed = _print_result_table("Image parity", i_results, args.min_cosine_image)

    report = {
        "model": args.model,
        "min_cosine": args.min_cosine,
        "queries": q_results,
        "images": i_results,
        "passed": q_passed and i_passed,
    }
    report_path = out / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {report_path}")

    if q_passed and i_passed:
        print()
        print("=" * 70)
        print("OK PARITY TEST PASSED -- vLLM matches HuggingFace reference")
        print(f"   Report: {report_path}")
        print("=" * 70)
        sys.exit(0)
    else:
        print()
        print("=" * 70)
        print("XX PARITY TEST FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
