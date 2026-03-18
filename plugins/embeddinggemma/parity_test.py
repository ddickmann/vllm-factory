"""
EmbeddingGemma Parity Test — vLLM plugin vs SentenceTransformers

Two-phase design (reference venv vs main venv):
  Phase 1 (--prepare): SentenceTransformers reference embeddings → saved to disk
  Phase 2 (--test):    vLLM inference + cosine similarity comparison

Usage:
    python plugins/embeddinggemma/parity_test.py --prepare
    python plugins/embeddinggemma/parity_test.py --test
    python plugins/embeddinggemma/parity_test.py          # both (same env)
"""

import argparse
import os
import subprocess
import sys
import time

import torch

MODEL = "unsloth/embeddinggemma-300m"
REF_FILE = "/tmp/embeddinggemma-reference.pt"

TEXTS = [
    "task: search result | query: What is machine learning?",
    "task: search result | query: How does gradient descent work?",
    "title: none | text: Machine learning is a subset of artificial intelligence that "
    "enables systems to learn and improve from experience without being explicitly programmed.",
    "task: sentence similarity | query: The quick brown fox jumps over the lazy dog.",
    "task: clustering | query: Advances in natural language processing have led to "
    "significant improvements in text understanding and generation capabilities.",
]


def phase_prepare():
    """Generate SentenceTransformers reference embeddings."""
    from sentence_transformers import SentenceTransformer

    print("=" * 60)
    print("PHASE 1: SentenceTransformers Reference")
    print("=" * 60)

    model = SentenceTransformer(MODEL, trust_remote_code=True)

    embeddings = model.encode(TEXTS, convert_to_tensor=True, normalize_embeddings=True)
    print(f"Reference embeddings: {embeddings.shape}")

    torch.save({
        "embeddings": embeddings.cpu().float(),
        "texts": TEXTS,
        "model": MODEL,
    }, REF_FILE)
    print(f"Saved to {REF_FILE}")
    print("Phase 1 complete\n")


def phase_test():
    """Run vLLM inference and compare with saved references."""
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    import plugins.embeddinggemma  # noqa: F401

    print("=" * 60)
    print("PHASE 2: vLLM Inference + Parity")
    print("=" * 60)

    ref = torch.load(REF_FILE, weights_only=False)
    ref_embeddings = ref["embeddings"]
    print(f"Reference: {ref_embeddings.shape}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="float32",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.4,
    )

    inputs = []
    for text in TEXTS:
        tokens = tokenizer(text, truncation=True, max_length=2048, return_tensors=None)
        inputs.append(TokensPrompt(prompt_token_ids=tokens["input_ids"]))

    _ = llm.embed(inputs)
    N = 5
    t0 = time.perf_counter()
    for _ in range(N):
        outputs = llm.embed(inputs)
    latency = (time.perf_counter() - t0) / N * 1000

    vllm_embeddings = torch.stack([
        torch.as_tensor(o.outputs.embedding).float() for o in outputs
    ])
    print(f"vLLM embeddings: {vllm_embeddings.shape}, Latency: {latency:.1f}ms")

    # Cosine similarity per sample
    print(f"\n{'─'*60}")
    print("  Parity (cosine similarity, threshold >= 0.99)")
    print(f"{'─'*60}")

    all_passed = True
    cosines = []
    for i, (ref_emb, vllm_emb) in enumerate(zip(ref_embeddings, vllm_embeddings)):
        cos = torch.nn.functional.cosine_similarity(
            ref_emb.unsqueeze(0), vllm_emb.unsqueeze(0)
        ).item()
        cosines.append(cos)
        passed = cos >= 0.99
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        text_short = TEXTS[i][:45]
        print(f"  {i}: cos={cos:.6f}  {status}  {text_short}...")

    mean_cos = sum(cosines) / len(cosines)
    print(f"{'─'*60}")
    print(f"  Mean cosine: {mean_cos:.6f}")
    print(f"  Latency: {latency:.1f}ms for {len(TEXTS)} texts")

    if all_passed:
        print("PARITY OK")
    else:
        print("NEEDS WORK")

    del llm
    torch.cuda.empty_cache()
    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmbeddingGemma Parity Test")
    parser.add_argument("--prepare", action="store_true", help="Phase 1: generate references")
    parser.add_argument("--test", action="store_true", help="Phase 2: vLLM inference + comparison")
    args = parser.parse_args()

    if args.prepare:
        phase_prepare()
    elif args.test:
        sys.exit(phase_test())
    else:
        print("Running both phases in separate processes...\n")
        r1 = subprocess.run([sys.executable, __file__, "--prepare"], cwd=os.getcwd())
        if r1.returncode != 0:
            sys.exit(r1.returncode)
        r2 = subprocess.run([sys.executable, __file__, "--test"], cwd=os.getcwd())
        sys.exit(r2.returncode)
