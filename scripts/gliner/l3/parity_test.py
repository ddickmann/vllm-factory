#!/usr/bin/env python3
"""
GLiNER-Linker Full Pipeline Parity Test.

Phase 1: Generate reference entity predictions using HuggingFace DebertaModel
          (standalone, no GLiNER library — mirrors the exact model forward pass)
Phase 2: Run vLLM pipeline and compare token-level logits

Runs each phase in a separate process to avoid GPU memory conflicts.

Usage:
    cd vllm-factory && PYTHONPATH=. python scripts/gliner/l3/parity_test.py [--regen]
"""

import json
import multiprocessing as mp
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HF_MODEL = "knowledgator/gliner-linker-large-v1.0"
REF_FILE = "/tmp/glinker-full-reference.pt"


def _resolve_model_path() -> str:
    """Resolve the local model directory path (auto-generates if needed)."""
    from plugins.deberta_gliner_linker import get_model_path
    return get_model_path()

# Test data
TEST_TEXT = "Apple announced new products in California. Michael Jordan joined the team."
TEST_LABELS = ["company", "location", "person"]


def phase_ref():
    """Phase 1: Generate reference logits using HuggingFace DebertaModel."""
    import re

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, DebertaConfig, DebertaModel

    print("=" * 60)
    print("PHASE 1: HuggingFace Reference (Full Pipeline)")
    print("=" * 60)

    # Load config
    cfg_path = hf_hub_download(HF_MODEL, "gliner_config.json")
    with open(cfg_path) as f:
        gliner_cfg = json.load(f)

    enc_cfg = gliner_cfg["encoder_config"]
    le_cfg = gliner_cfg["labels_encoder_config"]

    # Build text encoder
    text_deberta_cfg = DebertaConfig(**enc_cfg)
    text_model = DebertaModel(text_deberta_cfg)

    # Build labels encoder
    labels_deberta_cfg = DebertaConfig(**le_cfg)
    labels_model = DebertaModel(labels_deberta_cfg)

    # Load weights
    weights_path = hf_hub_download(HF_MODEL, "pytorch_model.bin")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)

    text_prefix = "token_rep_layer.bert_layer.model."
    labels_prefix = "token_rep_layer.labels_encoder.model."

    text_state = {k[len(text_prefix):]: v for k, v in state.items() if k.startswith(text_prefix)}
    labels_state = {k[len(labels_prefix):]: v for k, v in state.items() if k.startswith(labels_prefix)}

    text_model.load_state_dict(text_state, strict=False)
    labels_model.load_state_dict(labels_state, strict=False)
    print(f"Text encoder: {len(text_state)} keys")
    print(f"Labels encoder: {len(labels_state)} keys")

    # Load scorer (BiEncoderTokenModel does not apply the checkpoint LSTM before scorer)
    H = enc_cfg["hidden_size"]
    scorer_proj_token = nn.Linear(H, H * 2)
    scorer_proj_label = nn.Linear(H, H * 2)
    scorer_out_mlp = nn.Sequential(
        nn.Linear(H * 3, H * 4),
        nn.Dropout(0.0),
        nn.ReLU(),
        nn.Linear(H * 4, 3),
    )

    scorer_proj_token.load_state_dict({
        k[len("scorer.proj_token."):]: v
        for k, v in state.items() if k.startswith("scorer.proj_token.")
    })
    scorer_proj_label.load_state_dict({
        k[len("scorer.proj_label."):]: v
        for k, v in state.items() if k.startswith("scorer.proj_label.")
    })

    mlp_state = {}
    for k, v in state.items():
        if k.startswith("scorer.out_mlp."):
            key = k[len("scorer.out_mlp."):]
            mlp_state[key] = v
    scorer_out_mlp.load_state_dict(mlp_state)
    print("Scorer loaded")

    # Move to GPU
    device = "cuda"
    text_model.eval().to(device)
    labels_model.eval().to(device)
    scorer_proj_token.eval().to(device)
    scorer_proj_label.eval().to(device)
    scorer_out_mlp.eval().to(device)

    # 1. Tokenize text (using transformer_tokenizer with is_split_into_words=True)
    tokenizer = AutoTokenizer.from_pretrained(_resolve_model_path(), use_fast=True)

    word_pattern = re.compile(r'\w+(?:[-_]\w+)*|\S')
    words = [m.group() for m in word_pattern.finditer(TEST_TEXT)]
    word_positions = [(m.start(), m.end()) for m in word_pattern.finditer(TEST_TEXT)]
    print(f"\nText: '{TEST_TEXT}'")
    print(f"Words ({len(words)}): {words}")

    # Tokenize as pre-split words
    tok_result = tokenizer(
        [words],
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="longest",
    )
    input_ids = tok_result["input_ids"].to(device)
    attention_mask = tok_result["attention_mask"].to(device)

    # Build words_mask using word_ids()
    word_ids_list = tok_result.word_ids(batch_index=0)
    words_mask = torch.zeros(len(word_ids_list), dtype=torch.long, device=device)
    prev_wid = -1
    for idx, wid in enumerate(word_ids_list):
        if wid is not None and wid != prev_wid:
            words_mask[idx] = wid + 1  # 1-indexed
            prev_wid = wid

    text_lengths = torch.tensor([len(words)], device=device)

    print(f"input_ids shape: {input_ids.shape}")
    print(f"words_mask nonzero: {(words_mask > 0).sum().item()}")
    print(f"text_lengths: {text_lengths.tolist()}")

    # 2. Encode text
    with torch.no_grad():
        text_output = text_model(input_ids=input_ids, attention_mask=attention_mask)
    text_hidden = text_output.last_hidden_state  # (1, L, H)

    # 3. Extract word embeddings
    W = int(text_lengths.max().item())
    embed_dim = text_hidden.shape[-1]
    word_embs = torch.zeros(1, W, embed_dim, dtype=text_hidden.dtype, device=device)

    # Extract using words_mask
    batch_idx, token_pos = torch.where(words_mask.unsqueeze(0) > 0)
    word_target = (words_mask.unsqueeze(0)[batch_idx, token_pos] - 1).long()
    # GLiNER bi-encoder checkpoints use subtoken_pooling="first".
    keep = torch.ones_like(word_target, dtype=torch.bool)
    if word_target.numel() > 1:
        keep[1:] = word_target[1:] != word_target[:-1]
    word_embs[batch_idx[keep], word_target[keep]] = text_hidden[batch_idx[keep], token_pos[keep]]

    print(f"Word embeddings shape: {word_embs.shape}")

    # 4. Encode labels
    labels_tokenizer = AutoTokenizer.from_pretrained(_resolve_model_path(), use_fast=True)
    all_label_embs = []
    for label in TEST_LABELS:
        enc = labels_tokenizer(label, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            le_out = labels_model(input_ids=enc["input_ids"].to(device))
        hs = le_out.last_hidden_state
        mask_expanded = enc["attention_mask"].to(device).unsqueeze(-1).expand(hs.size()).float()
        mean = (hs * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        all_label_embs.append(mean.squeeze(0))

    label_embs = torch.stack(all_label_embs, dim=0).unsqueeze(0)  # (1, C, H)
    print(f"Label embeddings shape: {label_embs.shape}")

    # 5. Scorer
    B, W_actual, H_actual = word_embs.shape
    C = label_embs.shape[1]

    token_rep = scorer_proj_token(word_embs).view(B, W_actual, 1, 2, H_actual)
    label_rep = scorer_proj_label(label_embs).view(B, 1, C, 2, H_actual)

    token_rep = token_rep.expand(-1, -1, C, -1, -1).permute(3, 0, 1, 2, 4)
    label_rep = label_rep.expand(-1, W_actual, -1, -1, -1).permute(3, 0, 1, 2, 4)

    cat = torch.cat([token_rep[0], label_rep[0], token_rep[1] * label_rep[1]], dim=-1)

    with torch.no_grad():
        scores = scorer_out_mlp(cat)  # (1, W, C, 3)
    scores = scores.squeeze(0)  # (W, C, 3)

    print(f"\nScores shape: {scores.shape}")
    print(f"Scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")

    # Decode entities from scores
    scores_sigmoid = torch.sigmoid(scores)
    threshold = 0.5
    # Channel 0 = start, channel 1 = end, channel 2 = inside
    start_scores = scores_sigmoid[:, :, 0]  # (W, C)
    end_scores = scores_sigmoid[:, :, 1]    # (W, C)
    inside_scores = scores_sigmoid[:, :, 2]  # (W, C)

    print("\nPer-label max start scores:")
    for c, label in enumerate(TEST_LABELS):
        max_val = start_scores[:, c].max().item()
        max_pos = start_scores[:, c].argmax().item()
        print(f"  {label}: max={max_val:.4f} at word '{words[max_pos]}' (idx={max_pos})")

    # Find entities (BIO-style)
    {i + 1: label for i, label in enumerate(TEST_LABELS)}
    entities = []
    for c in range(C):
        starts = (start_scores[:, c] > threshold).nonzero(as_tuple=False).flatten()
        ends = (end_scores[:, c] > threshold).nonzero(as_tuple=False).flatten()
        for st in starts:
            for ed in ends:
                if ed >= st:
                    ins = inside_scores[st:ed+1, c]
                    if (ins < threshold).any():
                        continue
                    combined = torch.cat([ins, start_scores[st, c:c+1], end_scores[ed, c:c+1]])
                    score = combined.min().item()
                    entity_text = " ".join(words[st:ed+1])
                    entities.append({
                        "text": entity_text,
                        "label": TEST_LABELS[c],
                        "score": score,
                        "start": st.item(),
                        "end": ed.item(),
                    })

    print(f"\n--- Entities (threshold={threshold}) ---")
    for e in entities:
        print(f"  '{e['text']}' → {e['label']} (score={e['score']:.4f}, words {e['start']}-{e['end']})")

    if not entities:
        print("  (no entities found — try lower threshold)")
        # Show top predictions for each label
        for c, label in enumerate(TEST_LABELS):
            top_start = start_scores[:, c].topk(3)
            print(f"  {label} top starts: {[(words[i], f'{v:.4f}') for i, v in zip(top_start.indices.tolist(), top_start.values.tolist())]}")

    # Save everything for Phase 2 comparison
    torch.save({
        "scores": scores.cpu(),
        "words": words,
        "word_positions": word_positions,
        "labels": TEST_LABELS,
        "entities": entities,
        "text": TEST_TEXT,
        "words_mask": words_mask.cpu(),
        "input_ids": input_ids[0].cpu(),
        "text_lengths": text_lengths.cpu(),
        "label_embs": label_embs.squeeze(0).cpu(),  # (C, H)
    }, REF_FILE)
    print(f"\nSaved reference to {REF_FILE}")


def phase_test():
    """Phase 2: vLLM full pipeline test."""
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    from vllm.pooling_params import PoolingParams

    print("=" * 60)
    print("PHASE 2: vLLM Full Pipeline")
    print("=" * 60)

    ref = torch.load(REF_FILE, map_location="cpu", weights_only=True)
    ref_scores = ref["scores"]
    words = ref["words"]
    labels = ref["labels"]
    input_ids_ref = ref["input_ids"]
    words_mask_ref = ref["words_mask"]
    text_lengths_ref = ref["text_lengths"]
    label_embs_ref = ref["label_embs"]

    print(f"Ref scores shape: {ref_scores.shape}")
    print(f"Words: {words}")
    print(f"Labels: {labels}")

    # Create LLM
    llm = LLM(
        model=_resolve_model_path(),
        trust_remote_code=True,
        dtype="float32",
        max_model_len=512,
        enforce_eager=True,
        disable_log_stats=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.35,
    )

    # Prepare input with extra_kwargs for the pooler
    gliner_data = {
        "input_ids": input_ids_ref.tolist(),
        "words_mask": words_mask_ref.tolist(),
        "text_lengths": int(text_lengths_ref[0].item()),
        "threshold": 0.5,
        "labels_embeds": label_embs_ref.tolist(),  # Pass precomputed label embeddings
    }

    prompt = TokensPrompt(prompt_token_ids=input_ids_ref.tolist())
    pooling_params = PoolingParams(task="plugin", extra_kwargs=gliner_data)

    print("\n--- Running vLLM inference ---")
    t0 = time.perf_counter()
    outputs = llm.encode(
        [prompt],
        pooling_params=pooling_params,
        pooling_task="plugin",
    )
    latency = (time.perf_counter() - t0) * 1000
    print(f"Latency: {latency:.1f}ms")

    # Decode output
    raw = torch.as_tensor(outputs[0].outputs.data)
    print(f"Raw output shape: {raw.shape}")

    if raw.numel() < 4:
        print("❌ Output too small — pooler returned dummy")
        return

    # Parse shape prefix
    W = int(raw[0].item())
    C = int(raw[1].item())
    S = int(raw[2].item())
    N = int(raw[3].item()) if raw.numel() > 4 else 0
    expected = 4 + (W * C * S) + (N * 2) + N + (N * C)
    if expected == raw.numel():
        vllm_scores = raw[4 : 4 + (W * C * S)].reshape(W, C, S)
    else:
        vllm_scores = raw[3:].reshape(W, C, S)

    print(f"vLLM scores shape: {vllm_scores.shape}")
    print(f"Ref scores shape: {ref_scores.shape}")

    # Compare
    if vllm_scores.shape != ref_scores.shape:
        print(f"❌ Shape mismatch! vLLM={vllm_scores.shape}, ref={ref_scores.shape}")
        return

    # Cosine similarity on the full flattened scores
    cos_sim = F.cosine_similarity(
        vllm_scores.flatten().unsqueeze(0),
        ref_scores.flatten().unsqueeze(0)
    ).item()

    # Element-wise comparison
    max_diff = (vllm_scores - ref_scores).abs().max().item()
    mean_diff = (vllm_scores - ref_scores).abs().mean().item()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"cos_sim (flat):  {cos_sim:.6f}")
    print(f"max_diff:        {max_diff:.6f}")
    print(f"mean_diff:       {mean_diff:.6f}")
    print(f"Latency:         {latency:.1f}ms")

    if cos_sim >= 0.999 and max_diff < 0.01:
        print("✅ PARITY OK")
    elif cos_sim >= 0.99:
        print("⚠️  CLOSE — needs investigation")
    else:
        print("❌ PARITY FAILED")


if __name__ == "__main__":
    # Verify model access
    model_path = _resolve_model_path()
    print(f"Using model dir: {model_path}")

    regen = "--regen" in sys.argv

    if regen or not os.path.exists(REF_FILE):
        print("Generating reference...\n")
        p1 = mp.Process(target=phase_ref)
        p1.start()
        p1.join()
    else:
        print(f"Using existing reference: {REF_FILE}")
        print("(pass --regen to regenerate)\n")

    print("\nRunning both phases in separate processes...\n")

    # Phase 2 in subprocess
    p2 = mp.Process(target=phase_test)
    p2.start()
    p2.join()

