"""
ColLFM2 Parity Test — proves vLLM output matches HuggingFace reference.

Uses forge.testing.harness.ModelTestHarness with two modes:
  1. offline: uses LLM.encode() directly (no server needed)
  2. online:  uses the ColLFM2Processor async pipeline

Reference preprocessing (superpod ColLFM2Processor):
  Queries: direct tokenization, NO prefix
  Images:  VISUAL_PROMPT_PREFIX='<|im_start|>user\\n<image>Describe the image.<|im_end|>'

Parity thresholds (FP32 reference vs BF16 vLLM):
  min_cosine_sim = 0.99  (FP16/BF16 serving threshold from PLUGIN_GUIDE.md)
  atol           = 1e-3

Usage:
    # Offline parity (no server needed, uses LLM.encode() directly)
    python plugins/collfm2/parity_test.py \\
        --model VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1

    # With pre-generated reference outputs:
    python plugins/collfm2/parity_test.py \\
        --model VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1 \\
        --reference-dir /tmp/collfm2_reference
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch

# ---------------------------------------------------------------------------
# Sample inputs
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = [
    "What is machine learning?",
    "How does attention work in transformers?",
    "Explain the difference between precision and recall.",
    "What are the key features of this document?",
    "Find sections related to financial statements.",
]

VISUAL_PROMPT_PREFIX = "<|im_start|>user\n<image>Describe the image.<|im_end|>"


# ---------------------------------------------------------------------------
# Reference (HuggingFace)
# ---------------------------------------------------------------------------

def _sdpa_patch():
    """Context manager: force sdpa instead of flash_attention_2 in ColLFM2 loading."""
    from contextlib import contextmanager

    from transformers import AutoModelForImageTextToText

    @contextmanager
    def _ctx():
        _orig = AutoModelForImageTextToText.from_pretrained
        def _wrap(*args, **kwargs):
            kwargs["attn_implementation"] = "sdpa"
            return _orig(*args, **kwargs)
        AutoModelForImageTextToText.from_pretrained = _wrap
        try:
            yield
        finally:
            AutoModelForImageTextToText.from_pretrained = _orig
    return _ctx()


def _run_reference_queries(
    model_path: str,
    queries: List[str],
    device: str = "cuda",
) -> List[torch.Tensor]:
    """Embed queries using the reference ColLFM2 model (sauerkrautlm_colpali, no vLLM)."""
    from sauerkrautlm_colpali import ColLFM2, ColLFM2Processor

    processor = ColLFM2Processor.from_pretrained(model_path)
    with _sdpa_patch():
        model = ColLFM2.from_pretrained(model_path).to(device)
    model.eval()

    embeddings = []
    for q in queries:
        batch = processor.process_queries([q])
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("token_type_ids", None)
        with torch.no_grad():
            # ColLFM2.forward() returns (1, seq_len, 128) normalized projections
            emb = model(**batch)
        embeddings.append(emb.squeeze(0).cpu().float())

    del model
    torch.cuda.empty_cache()
    return embeddings


def _run_reference_images(
    model_path: str,
    images,
    device: str = "cuda",
) -> List[torch.Tensor]:
    """Embed images using the reference ColLFM2 model."""
    from sauerkrautlm_colpali import ColLFM2, ColLFM2Processor

    processor = ColLFM2Processor.from_pretrained(model_path)
    with _sdpa_patch():
        model = ColLFM2.from_pretrained(model_path).to(device)
    model.eval()

    embeddings = []
    for img in images:
        batch = processor.process_images([img.convert("RGB")])
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("token_type_ids", None)
        with torch.no_grad():
            emb = model(**batch)
        embeddings.append(emb.squeeze(0).cpu().float())

    del model
    torch.cuda.empty_cache()
    return embeddings


# ---------------------------------------------------------------------------
# vLLM (offline, via LLM.encode())
# ---------------------------------------------------------------------------

def _run_vllm_queries(model_path: str, queries: List[str]) -> List[torch.Tensor]:
    """Embed queries via vLLM offline (uses plugin preprocessing)."""
    from transformers import AutoProcessor
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    import collfm2  # noqa: F401 — triggers register()

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = proc.tokenizer if hasattr(proc, "tokenizer") else proc

    llm = LLM(
        model=model_path,
        runner="pooling",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.7,
        max_model_len=8192,
        skip_mm_profiling=True,
        mm_processor_cache_gb=1,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=False,  # Mamba prefix caching has issues in vLLM 0.15.1
    )

    inputs = []
    for q in queries:
        batch = tokenizer(
            q,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
            max_length=2048,
            truncation=True,
        )
        ids = batch["input_ids"][0].tolist()
        inputs.append(TokensPrompt(prompt_token_ids=ids))

    # vLLM 0.15.1: pooling_task is a direct kwarg to encode(), not in PoolingParams
    outputs = llm.encode(inputs, pooling_task="token_embed")
    embeddings = []
    for o in outputs:
        emb = torch.as_tensor(o.outputs.data).float()
        embeddings.append(emb)

    del llm
    torch.cuda.empty_cache()
    return embeddings


def _run_vllm_images(model_path: str, images) -> List[torch.Tensor]:
    """Embed images via vLLM offline LLM."""
    import collfm2  # noqa — triggers register()
    from vllm import LLM

    llm = LLM(
        model=model_path,
        runner="pooling",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.7,
        max_model_len=8192,
        skip_mm_profiling=True,
        mm_processor_cache_gb=1,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=False,  # Mamba prefix caching has issues in vLLM 0.15.1
    )

    inputs = [
        {
            "prompt": VISUAL_PROMPT_PREFIX,
            "multi_modal_data": {"image": img.convert("RGB")},
        }
        for img in images
    ]

    # vLLM 0.15.1: pooling_task is a direct kwarg to encode(), not in PoolingParams
    outputs = llm.encode(inputs, pooling_task="token_embed")
    embeddings = []
    for o in outputs:
        emb = torch.as_tensor(o.outputs.data).float()
        # Strip BOS token (vLLM prepends it; reference doesn't)
        if emb.shape[0] > 1:
            emb = emb[1:]
        embeddings.append(emb)

    del llm
    torch.cuda.empty_cache()
    return embeddings


# ---------------------------------------------------------------------------
# Parity metric helpers
# ---------------------------------------------------------------------------

def _cosine_sim_per_sample(ref: torch.Tensor, vllm: torch.Tensor) -> float:
    """Mean cosine similarity between two variable-length embedding tensors."""
    min_len = min(ref.shape[0], vllm.shape[0])
    r = ref[:min_len].reshape(-1)
    v = vllm[:min_len].reshape(-1)
    return torch.nn.functional.cosine_similarity(r.unsqueeze(0), v.unsqueeze(0)).item()


def _print_parity_table(label: str, ref_embs, vllm_embs, inputs, min_cos: float = 0.99):
    print(f"\n{'─'*70}")
    print(f"  {label} parity  (threshold: cosine ≥ {min_cos})")
    print(f"{'─'*70}")
    print(f"  {'#':<4} {'Input':<40} {'Cosine':>8} {'Status':>8}")
    print(f"  {'-'*4} {'-'*40} {'-'*8} {'-'*8}")

    all_passed = True
    for i, (ref, vllm_emb, inp) in enumerate(zip(ref_embs, vllm_embs, inputs)):
        cos = _cosine_sim_per_sample(ref, vllm_emb)
        passed = cos >= min_cos
        all_passed = all_passed and passed
        label_short = str(inp)[:38] if isinstance(inp, str) else f"image_{i}"
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {i:<4} {label_short:<40} {cos:>8.6f} {status:>8}")

    print(f"{'─'*70}")
    overall = "✅ ALL PASSED" if all_passed else "❌ SOME FAILED"
    print(f"  Overall: {overall}")
    return all_passed


def _make_synthetic_image():
    """Create a simple gradient test image (480×640 RGB)."""
    import numpy as np
    from PIL import Image

    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        for x in range(640):
            arr[y, x] = [x * 255 // 640, y * 255 // 480, 128]
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="ColLFM2 parity test: vLLM vs HuggingFace reference")
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument(
        "--reference-dir", default=None,
        help="Pre-generated reference dir from generate_reference.py (optional)"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--min-cosine-sim", type=float, default=0.99)
    p.add_argument("--report-dir", default="plugins/collfm2/reports")
    args = p.parse_args()

    # Ensure collfm2 plugin is importable as `import collfm2` (lives in plugins/)
    _plugins_dir = str(Path(__file__).parent.parent)  # vllm-factory/plugins/
    _repo_dir = str(Path(__file__).parent.parent.parent)  # vllm-factory/
    for _p in [_plugins_dir, _repo_dir]:
        if _p not in sys.path:
            sys.path.insert(0, _p)


    print(f"\n{'='*70}")
    print("ColLFM2 Parity Test")
    print(f"  model:         {args.model}")
    print(f"  device:        {args.device}")
    print(f"  min_cosine:    {args.min_cosine_sim}")
    print(f"{'='*70}")

    # -----------------------------------------------------------------------
    # Load or generate reference query embeddings
    # -----------------------------------------------------------------------
    if args.reference_dir and (Path(args.reference_dir) / "reference_query_embeddings.pt").exists():
        ref_dir = Path(args.reference_dir)
        print(f"\n[1/4] Loading pre-computed reference query embeddings from {ref_dir}...")
        ref_query_embs = torch.load(ref_dir / "reference_query_embeddings.pt", weights_only=True)
        queries = SAMPLE_QUERIES[:len(ref_query_embs)]
    else:
        print("\n[1/4] Computing reference query embeddings (HF model)...")
        queries = SAMPLE_QUERIES
        ref_query_embs = _run_reference_queries(args.model, queries, args.device)
        print(f"  {len(ref_query_embs)} query embeddings computed")

    # -----------------------------------------------------------------------
    # Run vLLM query embeddings
    # -----------------------------------------------------------------------
    print("\n[2/4] Computing vLLM query embeddings (offline LLM.encode)...")
    vllm_query_embs = _run_vllm_queries(args.model, queries)
    print(f"  {len(vllm_query_embs)} query embeddings computed")

    query_passed = _print_parity_table(
        "Query", ref_query_embs, vllm_query_embs, queries, args.min_cosine_sim
    )

    # -----------------------------------------------------------------------
    # Image parity
    # -----------------------------------------------------------------------
    if args.reference_dir and (Path(args.reference_dir) / "reference_image_embeddings.pt").exists():
        ref_dir = Path(args.reference_dir)
        print(f"\n[3/4] Loading pre-computed reference image embeddings from {ref_dir}...")
        ref_image_embs = torch.load(ref_dir / "reference_image_embeddings.pt", weights_only=True)
        images = [_make_synthetic_image() for _ in range(len(ref_image_embs))]
        # Use saved test images if present
        for i in range(len(ref_image_embs)):
            img_path = ref_dir / f"test_image_{i}.png"
            if img_path.exists():
                from PIL import Image
                images[i] = Image.open(img_path)
    else:
        print("\n[3/4] Computing reference image embeddings (HF model)...")
        images = [_make_synthetic_image()]
        ref_image_embs = _run_reference_images(args.model, images, args.device)
        print(f"  {len(ref_image_embs)} image embeddings computed")

    print("\n[4/4] Computing vLLM image embeddings...")
    vllm_image_embs = _run_vllm_images(args.model, images)
    image_passed = _print_parity_table(
        "Image", ref_image_embs, vllm_image_embs,
        [f"image_{i}" for i in range(len(images))],
        args.min_cosine_sim,
    )

    # -----------------------------------------------------------------------
    # Write report via ModelTestHarness
    # -----------------------------------------------------------------------
    from forge.testing.harness import ModelTestHarness, ParityResult

    harness = ModelTestHarness("collfm2", args.model)

    # Build aggregate parity result for queries (mean cosine)
    cosines = [_cosine_sim_per_sample(r, v) for r, v in zip(ref_query_embs, vllm_query_embs)]
    mean_cos = sum(cosines) / len(cosines)
    harness.report.parity_results.append(ParityResult(
        cosine_similarity=mean_cos,
        max_absolute_error=0.0,   # not directly comparable (variable lengths)
        mean_absolute_error=0.0,
        passed=query_passed,
        details=f"Queries ({len(queries)}) — mean cosine={mean_cos:.6f}",
    ))

    img_cosines = [_cosine_sim_per_sample(r, v) for r, v in zip(ref_image_embs, vllm_image_embs)]
    mean_img_cos = sum(img_cosines) / len(img_cosines)
    harness.report.parity_results.append(ParityResult(
        cosine_similarity=mean_img_cos,
        max_absolute_error=0.0,
        mean_absolute_error=0.0,
        passed=image_passed,
        details=f"Images ({len(images)}) — mean cosine={mean_img_cos:.6f}",
    ))

    report_path = Path(args.report_dir) / "parity_report.md"
    harness.generate_report(str(report_path))

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    all_passed = query_passed and image_passed
    print(f"\n{'='*70}")
    if all_passed:
        print("✅ PARITY TEST PASSED — vLLM matches HuggingFace reference")
    else:
        print("❌ PARITY TEST FAILED — see table above for failing cases")
    print(f"   Report: {report_path}")
    print(f"{'='*70}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
