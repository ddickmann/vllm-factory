"""
ColLFM2 Reference Output Generator

Generates ground-truth embeddings using the sauerkrautlm_colpali ColLFM2 model
(zero vLLM dependency) and saves them to disk for parity testing.

Uses the EXACT reference API from the sauerkrautlm-colpali package:
  - sauerkrautlm_colpali.ColLFM2           — model (downloads LiquidAI/LFM2-VL-450M base internally)
  - sauerkrautlm_colpali.ColLFM2Processor  — processor
  - process_queries()  → direct tokenization, NO prefix
  - process_images()   → VISUAL_PROMPT_PREFIX + dynamic image tokens
  - model.forward()    → returns (batch, seq_len, 128) normalized projections directly

Install dependency first:
    pip install git+https://github.com/VAGOsolutions/sauerkrautlm-colpali

Usage:
    python plugins/collfm2/generate_reference.py \\
        --model VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1 \\
        --output-dir /tmp/collfm2_reference

Outputs:
    <output-dir>/reference_query_embeddings.pt   — list of (seq_len, 128) tensors
    <output-dir>/reference_image_embeddings.pt   — list of (seq_len, 128) tensors
    <output-dir>/reference_metadata.json         — query strings, timings, req/s
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Reference query inputs
# ---------------------------------------------------------------------------
REFERENCE_QUERIES = [
    "What is machine learning?",
    "How does attention work in transformers?",
    "Explain the difference between precision and recall in information retrieval.",
    "What are the key features of this document?",
    "Find sections related to financial statements.",
    "What is the main conclusion of this research paper?",
    "Describe the methodology used in this study.",
    "What data sources were used?",
    "Who are the authors?",
    "What year was this published?",
]


def _make_synthetic_image(width: int = 640, height: int = 480) -> Image.Image:
    """Create a simple RGB gradient image for testing."""
    import numpy as np
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [x * 255 // width, y * 255 // height, 128]
    return Image.fromarray(arr, "RGB")


def _load_reference_model(model_path: str, device: str):
    """Load ColLFM2 model and processor from sauerkrautlm_colpali package.

    The ColLFM2 package hardcodes `attn_implementation='flash_attention_2'`
    internally. We monkeypatch AutoModelForImageTextToText.from_pretrained
    to override this to 'sdpa' (which works without flash_attn installed).
    """
    try:
        from sauerkrautlm_colpali import ColLFM2, ColLFM2Processor
    except ImportError:
        raise ImportError(
            "sauerkrautlm_colpali not installed.\n"
            "Run: pip install git+https://github.com/VAGOsolutions/sauerkrautlm-colpali"
        )

    # Patch: the package hardcodes flash_attention_2; we need sdpa (no flash_attn pkg)
    from transformers import AutoModelForImageTextToText
    _orig = AutoModelForImageTextToText.from_pretrained

    def _sdpa_from_pretrained(*args, **kwargs):
        kwargs["attn_implementation"] = "sdpa"
        return _orig(*args, **kwargs)

    AutoModelForImageTextToText.from_pretrained = _sdpa_from_pretrained

    try:
        print(f"[reference] Loading ColLFM2Processor from '{model_path}'...")
        processor = ColLFM2Processor.from_pretrained(model_path)

        print(f"[reference] Loading ColLFM2 model from '{model_path}' (attn=sdpa)...")
        model = ColLFM2.from_pretrained(model_path).to(device)
        model.eval()
    finally:
        AutoModelForImageTextToText.from_pretrained = _orig

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[reference] Model loaded ({n_params / 1e6:.1f}M params) on {device}")
    return model, processor




def _embed_queries_reference(
    queries: List[str],
    model,
    processor,
    device: str,
) -> List[torch.Tensor]:
    """
    Embed queries using exact reference preprocessing:
    - process_queries() → direct tokenization, NO prefix, padding="longest", max_length=2048
    - model() → returns (batch, seq_len, 128) normalized projections directly
    Returns list of per-query tensors [(seq_len_i, 128), ...]
    """
    all_embeddings = []
    for query in queries:
        # process_queries takes a list; process one at a time for variable-length outputs
        batch = processor.process_queries([query])
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("token_type_ids", None)

        with torch.no_grad():
            # ColLFM2.forward() returns (1, seq_len, 128) directly
            emb = model(**batch)

        # Squeeze batch dim → (seq_len, 128)
        emb = emb.squeeze(0).cpu().float()
        all_embeddings.append(emb)

    return all_embeddings


def _embed_images_reference(
    images: List[Image.Image],
    model,
    processor,
    device: str,
) -> List[torch.Tensor]:
    """
    Embed images using exact reference preprocessing:
    - process_images() → VISUAL_PROMPT_PREFIX text + dynamic image tokens
    - model() → returns (batch, seq_len, 128) normalized projections
    Returns list of per-image tensors [(seq_len_i, 128), ...]
    """
    all_embeddings = []
    for image in images:
        batch = processor.process_images([image.convert("RGB")])
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("token_type_ids", None)

        with torch.no_grad():
            emb = model(**batch)

        emb = emb.squeeze(0).cpu().float()
        all_embeddings.append(emb)

    return all_embeddings


def _benchmark_reference_throughput(
    queries: List[str],
    model,
    processor,
    device: str,
    n_runs: int = 50,
) -> dict:
    """Measure sequential throughput of the reference model (req/s baseline)."""

    # Warmup
    for q in queries[:3]:
        batch = processor.process_queries([q])
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("token_type_ids", None)
        with torch.no_grad():
            model(**batch)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    latencies = []
    test_queries = (queries * ((n_runs // len(queries)) + 1))[:n_runs]
    for q in test_queries:
        batch = processor.process_queries([q])
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("token_type_ids", None)

        t0 = time.perf_counter()
        with torch.no_grad():
            model(**batch)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    total = sum(latencies)
    sorted_lats = sorted(latencies)
    n = len(sorted_lats)
    return {
        "req_s": round(n_runs / total, 2),
        "p50_ms": round(sorted_lats[n // 2] * 1000, 1),
        "p95_ms": round(sorted_lats[int(n * 0.95)] * 1000, 1),
        "p99_ms": round(sorted_lats[int(n * 0.99)] * 1000, 1),
        "n_runs": n_runs,
        "device": device,
    }


def main():
    p = argparse.ArgumentParser(description="Generate reference outputs for ColLFM2 parity testing")
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--output-dir", default="/tmp/collfm2_reference", help="Output directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-images", type=int, default=3, help="Number of synthetic test images")
    p.add_argument("--n-bench-runs", type=int, default=50, help="Throughput benchmark iterations")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("ColLFM2 Reference Generator")
    print(f"  model:      {args.model}")
    print(f"  output_dir: {out_dir}")
    print(f"  device:     {args.device}")
    print(f"{'='*60}\n")

    model, processor = _load_reference_model(args.model, args.device)

    # -----------------------------------------------------------------------
    # 1. Generate query embeddings
    # -----------------------------------------------------------------------
    print("\n[1/3] Embedding reference queries...")
    t0 = time.perf_counter()
    query_embeddings = _embed_queries_reference(REFERENCE_QUERIES, model, processor, args.device)
    query_time = time.perf_counter() - t0

    print(f"  {len(query_embeddings)} queries in {query_time:.2f}s")
    for i, (q, emb) in enumerate(zip(REFERENCE_QUERIES, query_embeddings)):
        print(f"  [{i}] '{q[:50]}' → shape={tuple(emb.shape)}, norm={emb.norm(dim=-1).mean():.4f}")

    query_path = out_dir / "reference_query_embeddings.pt"
    torch.save(query_embeddings, query_path)
    print(f"  → Saved: {query_path}")

    # -----------------------------------------------------------------------
    # 2. Generate image embeddings
    # -----------------------------------------------------------------------
    print(f"\n[2/3] Embedding {args.n_images} synthetic test images...")
    images = [_make_synthetic_image(640, 480) for _ in range(args.n_images)]
    # Save test images for parity test reuse
    for i, img in enumerate(images):
        img.save(out_dir / f"test_image_{i}.png")

    t0 = time.perf_counter()
    image_embeddings = _embed_images_reference(images, model, processor, args.device)
    image_time = time.perf_counter() - t0

    print(f"  {len(image_embeddings)} images in {image_time:.2f}s")
    for i, emb in enumerate(image_embeddings):
        print(f"  [{i}] image_{i} → shape={tuple(emb.shape)}, norm={emb.norm(dim=-1).mean():.4f}")

    image_path = out_dir / "reference_image_embeddings.pt"
    torch.save(image_embeddings, image_path)
    print(f"  → Saved: {image_path}")

    # -----------------------------------------------------------------------
    # 3. Throughput benchmark (reference baseline)
    # -----------------------------------------------------------------------
    print(f"\n[3/3] Benchmarking reference throughput ({args.n_bench_runs} runs)...")
    bench = _benchmark_reference_throughput(
        REFERENCE_QUERIES, model, processor, args.device, args.n_bench_runs
    )
    print(f"  Reference baseline (ColLFM2 PyTorch, sequential, {args.device}):")
    print(f"    req/s: {bench['req_s']}")
    print(f"    p50:   {bench['p50_ms']} ms")
    print(f"    p95:   {bench['p95_ms']} ms")
    print(f"    p99:   {bench['p99_ms']} ms")

    # -----------------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------------
    metadata = {
        "model": args.model,
        "queries": REFERENCE_QUERIES,
        "n_queries": len(REFERENCE_QUERIES),
        "n_images": args.n_images,
        "query_embed_shapes": [list(e.shape) for e in query_embeddings],
        "image_embed_shapes": [list(e.shape) for e in image_embeddings],
        "reference_throughput": bench,
        "preprocessing": {
            "query": "process_queries(texts) — direct tokenization, NO prefix, padding=longest, max_length=2048",
            "image": "process_images(images) — VISUAL_PROMPT_PREFIX='<|im_start|>user\\n<image>Describe the image.<|im_end|>'",
            "note": "Uses sauerkrautlm_colpali.ColLFM2 + ColLFM2Processor",
        },
    }
    meta_path = out_dir / "reference_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  → Saved metadata: {meta_path}")

    print(f"\n{'='*60}")
    print("Reference generation complete.")
    print(f"  Queries:    {query_path}")
    print(f"  Images:     {image_path}")
    print(f"  Metadata:   {meta_path}")
    print(f"  Reference throughput: {bench['req_s']} req/s")
    print(f"{'='*60}\n")
    print("NEXT STEP: reinstall vllm==0.15.1 before running parity_test.py")
    print("  pip install vllm==0.15.1")


if __name__ == "__main__":
    main()
