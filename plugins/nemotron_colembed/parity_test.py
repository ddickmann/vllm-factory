"""
NemotronColEmbed Parity Test — nvidia/nemotron-colembed-vl-4b-v2

Two-phase design:
    Phase 1 (--prepare): HF reference embeddings for text queries + images
    Phase 2 (--test):    vLLM inference + cosine similarity comparison

Usage:
    python plugins/nemotron_colembed/parity_test.py --prepare
    python plugins/nemotron_colembed/parity_test.py --test
    python plugins/nemotron_colembed/parity_test.py  # both
"""

import argparse
import os
import subprocess
import sys
import time

import torch

MODEL = "nvidia/nemotron-colembed-vl-4b-v2"
REF_FILE = "/tmp/nemotron-colembed-reference.pt"

QUERIES = [
    "How is AI improving the intelligence and capabilities of robots?",
    "Canary, a multilingual model that transcribes speech in English, Spanish, German, and French.",
    "Generative AI can generate DNA sequences for bioengineering.",
]

IMAGE_URLS = [
    "https://developer.download.nvidia.com/images/isaac/nvidia-isaac-lab-1920x1080.jpg",
    "https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/asr-nemo-canary-featured.jpg",
    "https://blogs.nvidia.com/wp-content/uploads/2023/02/genome-sequencing-helix.jpg",
]


# ======================================================================
# Phase 1: HF reference embeddings
# ======================================================================

def phase_prepare():
    from transformers import AutoModel
    from transformers.image_utils import load_image

    print("=" * 60)
    print("PHASE 1: HF Reference Embeddings")
    print("=" * 60)

    model = AutoModel.from_pretrained(
        MODEL,
        device_map="cuda",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()

    # Query embeddings
    print("\nEncoding queries...")
    query_embs = model.forward_queries(QUERIES, batch_size=8)
    print(f"Query embeddings shape: {query_embs.shape}")  # (3, max_seq, hidden)

    # Image embeddings
    print("Loading & encoding images...")
    images = [load_image(url) for url in IMAGE_URLS]
    image_embs = model.forward_images(images, batch_size=8)
    print(f"Image embeddings shape: {image_embs.shape}")

    # Scores
    scores = model.get_scores(query_embs, image_embs)
    print(f"\nReference scores:\n{scores}")

    # Save reference
    torch.save({
        "query_embeddings": query_embs.cpu(),
        "image_embeddings": image_embs.cpu(),
        "scores": scores.cpu(),
        "queries": QUERIES,
        "image_urls": IMAGE_URLS,
    }, REF_FILE)
    print(f"\nSaved to {REF_FILE}")
    print("✅ Phase 1 complete\n")


# ======================================================================
# Phase 2: vLLM inference + cosine similarity comparison
# ======================================================================

def phase_test():
    from transformers.image_utils import load_image
    from vllm import LLM

    print("=" * 60)
    print("PHASE 2: vLLM Inference + Parity")
    print("=" * 60)

    # Load references
    ref = torch.load(REF_FILE, weights_only=False)
    ref_query_embs = ref["query_embeddings"]
    ref_image_embs = ref["image_embeddings"]
    ref_scores = ref["scores"]

    # Start vLLM
    print("\nStarting vLLM...")
    vllm_model = LLM(
        model=MODEL,
        runner="pooling",
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=4096,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
    )

    # ---- TEST 1: Text Query Embeddings ----
    print("\n--- TEST 1: Text Queries ---")

    # Format queries exactly like HF's process_queries:
    # 1. Prefix: "query: " (model.query_prefix)
    # 2. Wrap: "Query: {prefixed}" inside a chat template
    # 3. Apply chat_template with add_generation_prompt=True
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    formatted_queries = []
    for q in QUERIES:
        prefixed = f"query: {q}"  # query_prefix = "query:"
        message = [{"role": "user", "content": [{"type": "text", "text": f"Query: {prefixed}"}]}]
        formatted = proc.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        formatted_queries.append(formatted)
    print(f"  Formatted query 0: {repr(formatted_queries[0][:80])}...")

    query_inputs = [{"prompt": q} for q in formatted_queries]

    # Warmup
    _ = vllm_model.encode(query_inputs[:1], pooling_task="token_embed")

    # Timed run
    t0 = time.perf_counter()
    query_outputs = vllm_model.encode(query_inputs, pooling_task="token_embed")
    query_latency = (time.perf_counter() - t0) * 1000

    # Extract embeddings
    vllm_query_embs = [torch.as_tensor(o.outputs.data).float() for o in query_outputs]

    print(f"Query latency: {query_latency:.1f}ms for {len(QUERIES)} queries")
    print(f"vLLM query embedding shapes: {[e.shape for e in vllm_query_embs]}")

    # Compare query embeddings via cosine similarity
    query_sims = []
    for i, (vllm_emb, ref_emb) in enumerate(zip(vllm_query_embs, ref_query_embs)):
        # Get the non-padded ref embedding (remove zero-padding)
        ref_norms = ref_emb.norm(dim=-1)
        ref_mask = ref_norms > 0
        ref_valid = ref_emb[ref_mask].float()

        # Truncate vLLM to same length if needed
        vllm_valid = vllm_emb[:len(ref_valid)]

        if len(vllm_valid) > 0 and len(ref_valid) > 0:
            min_len = min(len(vllm_valid), len(ref_valid))
            vllm_valid = vllm_valid[:min_len]
            ref_valid = ref_valid[:min_len]
            # Per-token cosine similarity
            vn = vllm_valid / (vllm_valid.norm(dim=-1, keepdim=True) + 1e-9)
            rn = ref_valid / (ref_valid.norm(dim=-1, keepdim=True) + 1e-9)
            sim = (vn * rn).sum(dim=-1).mean().item()
        else:
            sim = 0.0
        query_sims.append(sim)
        print(f"  Query {i}: cos_sim = {sim:.6f}, "
              f"vllm_tokens={len(vllm_emb)}, ref_tokens={len(ref_valid)}")

    avg_query_sim = sum(query_sims) / len(query_sims) if query_sims else 0

    # ---- TEST 2: Image Embeddings ----
    print("\n--- TEST 2: Image Embeddings ---")

    images = [load_image(url) for url in IMAGE_URLS]

    # Build image prompts for vLLM multimodal input
    # Must match HF's process_documents format:
    #   passage_prefix = "passage:"
    #   text = f"{passage_prefix} {doc_text}" (empty text -> "passage: ")
    #   chat template with image + text, add_generation_prompt=True
    image_inputs = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Replicate process_documents formatting
        passage_text = "passage: "  # passage_prefix + " " + "" (empty text)
        message = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": passage_text},
        ]}]
        prompt = proc.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": img},
        })
    print(f"  Image prompt 0: {repr(image_inputs[0]['prompt'][:80])}...")

    t0 = time.perf_counter()
    image_outputs = vllm_model.encode(image_inputs, pooling_task="token_embed")
    image_latency = (time.perf_counter() - t0) * 1000

    vllm_image_embs = [torch.as_tensor(o.outputs.data).float() for o in image_outputs]

    print(f"Image latency: {image_latency:.1f}ms for {len(images)} images")
    print(f"vLLM image embedding shapes: {[e.shape for e in vllm_image_embs]}")

    # Compare image embeddings
    image_sims = []
    for i, (vllm_emb, ref_emb) in enumerate(zip(vllm_image_embs, ref_image_embs)):
        ref_norms = ref_emb.norm(dim=-1)
        ref_mask = ref_norms > 0
        ref_valid = ref_emb[ref_mask].float()
        vllm_valid = vllm_emb[:len(ref_valid)]

        if len(vllm_valid) > 0 and len(ref_valid) > 0:
            min_len = min(len(vllm_valid), len(ref_valid))
            vllm_valid = vllm_valid[:min_len]
            ref_valid = ref_valid[:min_len]
            vn = vllm_valid / (vllm_valid.norm(dim=-1, keepdim=True) + 1e-9)
            rn = ref_valid / (ref_valid.norm(dim=-1, keepdim=True) + 1e-9)
            sim = (vn * rn).sum(dim=-1).mean().item()
        else:
            sim = 0.0
        image_sims.append(sim)
        print(f"  Image {i}: cos_sim = {sim:.6f}, "
              f"vllm_tokens={len(vllm_emb)}, ref_tokens={len(ref_valid)}")

    avg_image_sim = sum(image_sims) / len(image_sims) if image_sims else 0

    # ---- TEST 3: MaxSim Scores ----
    print("\n--- TEST 3: MaxSim Scores ---")

    # Compute MaxSim between vLLM query and image embeddings
    vllm_scores = torch.zeros(len(QUERIES), len(images))
    for i, q_emb in enumerate(vllm_query_embs):
        for j, d_emb in enumerate(vllm_image_embs):
            # MaxSim: for each query token, find max similarity across doc tokens
            sim_matrix = torch.matmul(q_emb, d_emb.T)
            max_sim = sim_matrix.max(dim=1)[0].sum()
            vllm_scores[i, j] = max_sim

    print(f"Reference scores:\n{ref_scores}")
    print(f"\nvLLM scores:\n{vllm_scores}")

    # Check diagonal dominance (correct query matches correct image)
    ref_diag_correct = all(
        ref_scores[i].argmax() == i for i in range(min(len(QUERIES), len(images)))
    )
    vllm_diag_correct = all(
        vllm_scores[i].argmax() == i for i in range(min(len(QUERIES), len(images)))
    )

    print(f"\nRef diagonal dominant: {ref_diag_correct}")
    print(f"vLLM diagonal dominant: {vllm_diag_correct}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Avg query cos_sim:  {avg_query_sim:.6f}")
    print(f"Avg image cos_sim:  {avg_image_sim:.6f}")
    print(f"Query latency:      {query_latency:.1f}ms")
    print(f"Image latency:      {image_latency:.1f}ms")
    print(f"Diagonal dominant:  {'✅' if vllm_diag_correct else '❌'}")

    query_ok = avg_query_sim >= 0.90
    image_ok = avg_image_sim >= 0.90
    diag_ok = vllm_diag_correct

    if query_ok and image_ok and diag_ok:
        print("✅ PARITY OK")
    elif diag_ok:
        print("⚠️  Retrieval ranking correct but embedding similarity below threshold")
    else:
        print("❌ NEEDS WORK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NemotronColEmbed Parity Test")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        phase_prepare()
    elif args.test:
        phase_test()
    else:
        print("Running both phases in separate processes...\n")
        r1 = subprocess.run([sys.executable, __file__, "--prepare"], cwd=os.getcwd())
        if r1.returncode != 0:
            sys.exit(r1.returncode)
        r2 = subprocess.run([sys.executable, __file__, "--test"], cwd=os.getcwd())
        sys.exit(r2.returncode)
