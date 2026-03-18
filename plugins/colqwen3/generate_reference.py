"""
ColQwen3 Reference Generator

Generates HuggingFace reference embeddings using the sauerkrautlm-colpali
implementation. These are used as the ground truth for parity testing.

Usage:
    python plugins/colqwen3/generate_reference.py \
        --model VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1 \
        --output-dir /tmp/colqwen3_reference

Outputs (in output-dir):
    query_embeddings.pt     — list of (seq_len, 128) tensors
    image_embeddings.pt     — list of (seq_len, 128) tensors
    queries.json            — the query strings used
    images/                 — the test images saved as PNG
    hf_benchmark.json       — req/s, p50, p99 for HF baseline throughput
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

QUERIES = [
    "What is the main topic of this document?",
    "Summarise the key findings in one sentence.",
    "What methodology was used in this study?",
    "What are the conclusions drawn by the authors?",
    "List the main tables or figures described.",
]

# Synthetic test images (colored rectangles with text-like noise)
def _make_test_image(idx: int, size: tuple[int, int] = (1024, 1400)) -> Image.Image:
    """Create a synthetic document-like image."""
    import random
    random.seed(idx)
    colors = [
        (245, 245, 245), (240, 248, 255), (255, 248, 240),
    ]
    bg = colors[idx % len(colors)]
    img = Image.new("RGB", size, bg)

    # Add some colored rectangles to simulate text blocks
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    rng = random.Random(idx * 1337)
    for _ in range(30):
        x0 = rng.randint(50, size[0] - 200)
        y0 = rng.randint(30, size[1] - 50)
        w = rng.randint(80, 300)
        h = rng.randint(8, 16)
        gray = rng.randint(60, 180)
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(gray, gray, gray))

    return img


def _make_test_images(n: int = 3) -> list[Image.Image]:
    return [_make_test_image(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _time_hf_queries(model, processor, queries: list[str]) -> dict:
    """Measure HF inference throughput for queries."""
    # Warmup
    batch = processor.process_queries(queries[:1]).to(model.device)
    with torch.no_grad():
        _ = model(**batch)
    torch.cuda.synchronize()

    latencies = []
    for q in queries * 4:  # repeat for stable stats
        batch = processor.process_queries([q]).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**batch)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "req_s": round(1000 / statistics.median(latencies), 2),
        "p50_ms": round(statistics.median(latencies), 2),
        "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
    }


def _time_hf_images(model, processor, images: list[Image.Image]) -> dict:
    """Measure HF inference throughput for images."""
    # Warmup
    batch = processor.process_images(images[:1]).to(model.device)
    with torch.no_grad():
        _ = model(**batch)
    torch.cuda.synchronize()

    latencies = []
    for img in images * 4:
        batch = processor.process_images([img]).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**batch)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "req_s": round(1000 / statistics.median(latencies), 2),
        "p50_ms": round(statistics.median(latencies), 2),
        "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate ColQwen3 HF reference embeddings")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--output-dir", default="/tmp/colqwen3_reference")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-benchmark", action="store_true", help="Skip HF throughput benchmark")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("ColQwen3 Reference Generator")
    print(f"  model:  {args.model}")
    print(f"  output: {out}")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Load HuggingFace model (sauerkrautlm-colpali)
    # ------------------------------------------------------------------
    print("[1/4] Loading HuggingFace ColQwen3 model...")
    try:
        from sauerkrautlm_colpali.models import ColQwen3 as ColModel  # noqa: I001
        from sauerkrautlm_colpali.models import ColQwen3Processor as ColProcessor
    except ImportError:
        from sauerkrautlm_colpali.models import ColQwen2_5 as ColModel  # noqa: I001
        from sauerkrautlm_colpali.models import ColQwen2_5_Processor as ColProcessor

    # Patch: newer transformers moves hidden_size into text_config
    try:
        from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
        if not hasattr(Qwen2_5_VLConfig, "hidden_size"):
            Qwen2_5_VLConfig.hidden_size = property(
                lambda self: self.text_config.hidden_size
                if hasattr(self, "text_config") else 1536
            )
    except ImportError:
        pass

    model = ColModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=args.device,
    ).eval()

    processor = ColProcessor.from_pretrained(args.model)
    print("  ✓ Model loaded\n")

    # ------------------------------------------------------------------
    # Generate test images
    # ------------------------------------------------------------------
    images = _make_test_images(3)
    for i, img in enumerate(images):
        p_img = img_dir / f"image_{i}.png"
        img.save(p_img)
    print(f"[2/4] Generated {len(images)} test images → {img_dir}\n")

    # ------------------------------------------------------------------
    # Query embeddings
    # ------------------------------------------------------------------
    print("[3/4] Computing query embeddings...")
    query_embs: list[torch.Tensor] = []
    for q in QUERIES:
        batch = processor.process_queries([q]).to(model.device)
        with torch.no_grad():
            emb = model(**batch)
        query_embs.append(emb[0].cpu().float())
        print(f"  query '{q[:40]}...' → shape {emb[0].shape}")

    torch.save(query_embs, out / "query_embeddings.pt")
    with open(out / "queries.json", "w") as f:
        json.dump(QUERIES, f, indent=2)
    print(f"  ✓ Saved query_embeddings.pt ({len(query_embs)} tensors)\n")

    # ------------------------------------------------------------------
    # Image embeddings
    # ------------------------------------------------------------------
    print("[4/4] Computing image embeddings...")
    image_embs: list[torch.Tensor] = []
    for i, img in enumerate(images):
        batch = processor.process_images([img]).to(model.device)
        with torch.no_grad():
            emb = model(**batch)
        image_embs.append(emb[0].cpu().float())
        print(f"  image_{i} → shape {emb[0].shape}")

    torch.save(image_embs, out / "image_embeddings.pt")
    print(f"  ✓ Saved image_embeddings.pt ({len(image_embs)} tensors)\n")

    # ------------------------------------------------------------------
    # HF baseline benchmark
    # ------------------------------------------------------------------
    if not args.no_benchmark:
        print("[Benchmark] Measuring HF baseline throughput...")
        q_bench = _time_hf_queries(model, processor, QUERIES)
        i_bench = _time_hf_images(model, processor, images)
        bench = {"queries": q_bench, "images": i_bench}
        with open(out / "hf_benchmark.json", "w") as f:
            json.dump(bench, f, indent=2)
        print(f"  queries: {q_bench['req_s']} req/s  p50={q_bench['p50_ms']}ms  p99={q_bench['p99_ms']}ms")
        print(f"  images:  {i_bench['req_s']} req/s  p50={i_bench['p50_ms']}ms  p99={i_bench['p99_ms']}ms\n")

    print("=" * 70)
    print(f"✅ Reference outputs saved to {out}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
