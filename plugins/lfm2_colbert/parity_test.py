"""
LFM2-ColBERT Parity Test — vLLM plugin vs HuggingFace reference

Two-phase design:
  Phase 1 (--prepare): HF AutoModel reference embeddings → saved to disk
  Phase 2 (--test):    vLLM inference + cosine similarity comparison

No conflicting packages needed — Phase 1 uses only transformers + safetensors.

Usage:
    python plugins/lfm2_colbert/parity_test.py --prepare
    python plugins/lfm2_colbert/parity_test.py --test
    python plugins/lfm2_colbert/parity_test.py          # both
"""

import argparse
import os
import subprocess
import sys
import time

import torch

MODEL = "LiquidAI/LFM2-ColBERT-350M"
REF_FILE = "/tmp/lfm2-colbert-reference.pt"

QUERIES = [
    "What is machine learning?",
    "How does the Mamba architecture differ from transformers?",
    "Explain retrieval-augmented generation.",
]

DOCUMENTS = [
    "Machine learning is a branch of artificial intelligence that focuses on building "
    "systems that learn from data to improve their performance on specific tasks.",
    "The Mamba architecture replaces the attention mechanism with a selective state space "
    "model, enabling linear-time sequence processing with strong performance.",
    "Retrieval-augmented generation combines a retrieval component with a generative model "
    "to produce more accurate and grounded responses by conditioning on relevant documents.",
]


def phase_prepare():
    """Generate HuggingFace reference embeddings using AutoModel."""
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    print("=" * 60)
    print("PHASE 1: HF Reference Embeddings")
    print("=" * 60)

    print("Loading model...")
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Load projection layer
    proj_path = hf_hub_download(repo_id=MODEL, filename="1_Dense/model.safetensors")
    with safetensors.torch.safe_open(proj_path, framework="pt", device="cpu") as f:
        proj_weight = f.get_tensor("linear.weight").to(torch.bfloat16).cuda()

    def encode_texts(texts, hf_model, weight):
        embeddings = []
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=False)
            tokens = {k: v.cuda() for k, v in tokens.items()}
            with torch.no_grad():
                out = hf_model(**tokens)
            hidden = out.last_hidden_state.squeeze(0)  # (seq_len, hidden)
            projected = hidden @ weight.T  # (seq_len, 128)
            projected = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            embeddings.append(projected.cpu().float())
        return embeddings

    print("Encoding queries...")
    query_embs = encode_texts(QUERIES, model, proj_weight)
    print(f"  {len(query_embs)} queries, shapes: {[e.shape for e in query_embs]}")

    print("Encoding documents...")
    doc_embs = encode_texts(DOCUMENTS, model, proj_weight)
    print(f"  {len(doc_embs)} documents, shapes: {[e.shape for e in doc_embs]}")

    torch.save({
        "query_embeddings": query_embs,
        "document_embeddings": doc_embs,
        "queries": QUERIES,
        "documents": DOCUMENTS,
        "model": MODEL,
    }, REF_FILE)
    print(f"Saved to {REF_FILE}")
    print("Phase 1 complete\n")

    del model
    torch.cuda.empty_cache()


def phase_test():
    """Run vLLM inference and compare with saved references."""
    from transformers import AutoTokenizer
    from vllm import LLM

    import plugins.lfm2_colbert  # noqa: F401

    print("=" * 60)
    print("PHASE 2: vLLM Inference + Parity")
    print("=" * 60)

    ref = torch.load(REF_FILE, weights_only=False)
    ref_query_embs = ref["query_embeddings"]
    ref_doc_embs = ref["document_embeddings"]
    print(f"Reference: {len(ref_query_embs)} queries, {len(ref_doc_embs)} documents")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    llm = LLM(
        model=MODEL,
        runner="pooling",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=512,
        enable_prefix_caching=False,
    )

    def encode_texts(texts, engine):
        inputs = []
        for text in texts:
            tokens = tokenizer(text, truncation=True, max_length=512, return_tensors=None)
            inputs.append({"prompt_token_ids": tokens["input_ids"]})
        outputs = engine.encode(inputs, pooling_task="token_embed")
        return [torch.as_tensor(o.outputs.data).float() for o in outputs]

    # Warmup
    _ = encode_texts(QUERIES[:1], llm)

    t0 = time.perf_counter()
    vllm_query_embs = encode_texts(QUERIES, llm)
    vllm_doc_embs = encode_texts(DOCUMENTS, llm)
    latency = (time.perf_counter() - t0) * 1000

    def cosine(a, b):
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n].float(), b[:n].float()
        an = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
        bn = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
        return (an * bn).sum(dim=-1).mean().item()

    min_cos = 0.99
    print(f"\n{'─'*60}")
    print(f"  Query parity (cosine >= {min_cos})")
    print(f"{'─'*60}")

    all_passed = True
    for i, (ref_e, vllm_e) in enumerate(zip(ref_query_embs, vllm_query_embs)):
        cos = cosine(ref_e, vllm_e)
        passed = cos >= min_cos
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  query_{i}: cos={cos:.6f}  {status}  ref={ref_e.shape} vllm={vllm_e.shape}")

    print(f"\n{'─'*60}")
    print(f"  Document parity (cosine >= {min_cos})")
    print(f"{'─'*60}")

    for i, (ref_e, vllm_e) in enumerate(zip(ref_doc_embs, vllm_doc_embs)):
        cos = cosine(ref_e, vllm_e)
        passed = cos >= min_cos
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  doc_{i}:   cos={cos:.6f}  {status}  ref={ref_e.shape} vllm={vllm_e.shape}")

    print(f"\n{'─'*60}")
    print(f"  Latency: {latency:.1f}ms for {len(QUERIES) + len(DOCUMENTS)} texts")

    if all_passed:
        print("PARITY OK")
    else:
        print("NEEDS WORK")

    del llm
    torch.cuda.empty_cache()
    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LFM2-ColBERT Parity Test")
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
