"""
DeBERTa v2 GLiNER Parity Test — nvidia/gliner-PII

Two-phase design to avoid CUDA fork conflicts with vLLM V1:
  Phase 1 (--prepare): GLiNER reference + model dir preparation
  Phase 2 (--test):    vLLM inference + parity comparison

Usage:
    python plugins/deberta_gliner/parity_test.py --prepare   # generates references
    python plugins/deberta_gliner/parity_test.py --test      # runs vLLM + compares
    python plugins/deberta_gliner/parity_test.py             # both in sequence
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import torch

MODEL = "nvidia/gliner-PII"
LOCAL_MODEL_DIR = "/tmp/gliner-pii-vllm"
REF_FILE = "/tmp/gliner-pii-reference.json"

TEXT = (
    "Please contact John Smith at john.smith@example.com or call 555-123-4567. "
    "He lives at 123 Main Street, San Francisco, CA 94105. "
    "His social security number is 123-45-6789 and his credit card is 4532-1234-5678-9012. "
    "John works at NVIDIA Corporation in the AI research department."
)

LABELS = ["person", "email", "phone_number", "address", "organization"]
THRESHOLD = 0.5
MAX_WIDTH = 12
ENT_TOKEN = "<<ENT>>"
SEP_TOKEN = "<<SEP>>"
WORD_PATTERN = re.compile(r'\w+(?:[-_]\w+)*|\S')


# ======================================================================
# Phase 1: Generate references + model dir (no vLLM imports)
# ======================================================================

def phase_prepare():
    """Generate GLiNER reference entities and prepare vLLM model directory."""
    from gliner import GLiNER

    print("=" * 60)
    print("PHASE 1: GLiNER Reference + Model Directory")
    print("=" * 60)

    model = GLiNER.from_pretrained(MODEL)
    model.eval()

    entities = model.predict_entities(TEXT, LABELS, threshold=THRESHOLD)
    print(f"\nReference Entities ({len(entities)}):")
    for e in entities:
        print(f"  {e['text']:40s} => {e['label']:15s} (score: {e['score']:.4f})")

    # Save references
    with open(REF_FILE, "w") as f:
        json.dump({"entities": entities, "text": TEXT, "labels": LABELS, "threshold": THRESHOLD}, f, indent=2)
    print(f"\nSaved to {REF_FILE}")

    # Prepare vLLM model dir
    gliner_config = model.config.__dict__.copy()
    encoder_config = gliner_config.get("encoder_config", {})
    if hasattr(encoder_config, "__dict__"):
        encoder_config = encoder_config.__dict__

    tokenizer = model.data_processor.transformer_tokenizer
    ent_token_id = tokenizer.convert_tokens_to_ids(ENT_TOKEN)
    sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

    vllm_config = {
        "model_type": "gliner_deberta_v2",
        "architectures": ["GLiNERDebertaV2Model"],
        "num_hidden_layers": 0, "num_attention_heads": 1,
        "hidden_size": encoder_config.get("hidden_size", 1024),
        "vocab_size": encoder_config.get("vocab_size", 128004),
        "encoder_hidden_size": encoder_config.get("hidden_size", 1024),
        "encoder_num_hidden_layers": encoder_config.get("num_hidden_layers", 24),
        "encoder_num_attention_heads": encoder_config.get("num_attention_heads", 16),
        "encoder_intermediate_size": encoder_config.get("intermediate_size", 4096),
        "encoder_hidden_act": encoder_config.get("hidden_act", "gelu"),
        "encoder_hidden_dropout_prob": 0.0,
        "encoder_attention_probs_dropout_prob": 0.0,
        "encoder_max_position_embeddings": encoder_config.get("max_position_embeddings", 512),
        "encoder_type_vocab_size": encoder_config.get("type_vocab_size", 0),
        "encoder_layer_norm_eps": encoder_config.get("layer_norm_eps", 1e-7),
        "encoder_relative_attention": encoder_config.get("relative_attention", True),
        "encoder_max_relative_positions": encoder_config.get("max_relative_positions", -1),
        "encoder_position_buckets": encoder_config.get("position_buckets", 256),
        "encoder_pos_att_type": encoder_config.get("pos_att_type", ["p2c", "c2p"]),
        "encoder_share_att_key": encoder_config.get("share_att_key", True),
        "encoder_norm_rel_ebd": encoder_config.get("norm_rel_ebd", "layer_norm"),
        "encoder_position_biased_input": encoder_config.get("position_biased_input", False),
        "encoder_pad_token_id": encoder_config.get("pad_token_id", 0),
        "gliner_dropout": gliner_config.get("dropout", 0.4),
        "gliner_hidden_size": gliner_config.get("hidden_size", 512),
        "max_width": gliner_config.get("max_width", 12),
        "class_token_index": ent_token_id,
        "sep_token_index": sep_token_id,
        "has_rnn": gliner_config.get("has_rnn", True),
        "embed_ent_token": gliner_config.get("embed_ent_token", True),
        "max_len": gliner_config.get("max_len", 384),
        "span_mode": gliner_config.get("span_mode", "markerV0"),
        "subtoken_pooling": gliner_config.get("subtoken_pooling", "first"),
    }

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    with open(os.path.join(LOCAL_MODEL_DIR, "config.json"), "w") as f:
        json.dump(vllm_config, f, indent=2)

    state_dict = model.model.state_dict()
    deduped = {}
    seen = set()
    for k, v in state_dict.items():
        ptr = v.data_ptr()
        if ptr not in seen:
            deduped[k] = v.contiguous().cpu()
            seen.add(ptr)
        else:
            deduped[k] = v.clone().contiguous().cpu()

    import safetensors.torch
    safetensors.torch.save_file(deduped, os.path.join(LOCAL_MODEL_DIR, "model.safetensors"))
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    print(f"Model dir: {LOCAL_MODEL_DIR} ({len(deduped)} weights)")
    print("✅ Phase 1 complete\n")


# ======================================================================
# Phase 2: vLLM inference + parity comparison
# ======================================================================

def phase_test():
    """Run vLLM inference and compare with saved GLiNER references."""
    from transformers import AutoTokenizer
    from vllm import LLM, PoolingParams
    from vllm.inputs import TokensPrompt

    import plugins.deberta_gliner  # noqa: F401 — register before vLLM

    print("=" * 60)
    print("PHASE 2: vLLM Inference + Parity")
    print("=" * 60)

    # Load references
    with open(REF_FILE) as f:
        ref = json.load(f)
    ref_entities = ref["entities"]
    print(f"Reference: {len(ref_entities)} entities")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)

    # Tokenize
    words_with_pos = [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(TEXT)]
    words = [w[0] for w in words_with_pos]
    word_positions = [(w[1], w[2]) for w in words_with_pos]
    text_length = len(words)

    prompt_list = [token for label in LABELS for token in (ENT_TOKEN, label)]
    prompt_list.append(SEP_TOKEN)
    prompt_len = len(prompt_list)
    input_words = prompt_list + words

    tokenized = tokenizer(input_words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=False)
    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]

    # Words mask
    word_ids_list = tokenized.word_ids(batch_index=0)
    word_ids = torch.tensor([w if w is not None else -1 for w in word_ids_list], dtype=torch.long)
    prev_word_ids = torch.roll(word_ids, 1, dims=0)
    prev_word_ids[0] = -1
    is_new_word = (word_ids != -1) & (word_ids != prev_word_ids)
    is_in_text = (word_ids >= prompt_len)
    valid_indices = is_new_word & is_in_text
    words_mask = torch.zeros_like(word_ids)
    words_mask[valid_indices] = word_ids[valid_indices] - prompt_len + 1

    gliner_data = {
        "input_ids": input_ids.tolist(),
        "words_mask": words_mask.tolist(),
        "text_lengths": text_length,
        "attention_mask": attention_mask.tolist(),
    }

    prompt = TokensPrompt(prompt_token_ids=input_ids.tolist())
    pooling_params = PoolingParams(extra_kwargs=gliner_data)

    # Load vLLM
    vllm_model = LLM(
        model=LOCAL_MODEL_DIR,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
    )

    # Warmup + benchmark
    _ = vllm_model.embed([prompt], pooling_params=pooling_params)
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        outputs = vllm_model.embed([prompt], pooling_params=pooling_params)
    vllm_latency = (time.perf_counter() - t0) / N * 1000

    # Decode output
    raw = outputs[0].outputs.embedding
    scores = torch.tensor(raw)
    print(f"Output: {scores.shape}, Latency: {vllm_latency:.1f}ms")

    if scores.dim() == 1 and scores.numel() > 3:
        L, K_out, C = int(scores[0]), int(scores[1]), int(scores[2])
        logits = scores[3:].reshape(1, L, K_out, C)
    else:
        logits = scores.unsqueeze(0) if scores.dim() == 3 else scores

    id_to_classes = {i + 1: label for i, label in enumerate(LABELS)}
    vllm_entities = _decode_entities(logits, words, id_to_classes, word_positions, TEXT, THRESHOLD)

    # Compare
    print(f"\nvLLM Entities ({len(vllm_entities)}):")
    for e in vllm_entities:
        print(f"  {e['text']:40s} => {e['label']:15s} (score: {e['score']:.4f})")

    ref_set = {(e["text"], e["label"]) for e in ref_entities}
    vllm_set = {(e["text"], e["label"]) for e in vllm_entities}
    matched = ref_set & vllm_set
    precision = len(matched) / len(vllm_set) if vllm_set else 1
    recall = len(matched) / len(ref_set) if ref_set else 1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"Parity: P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    print(f"Reference: {len(ref_entities)}, vLLM: {len(vllm_entities)}, Matched: {len(matched)}")
    if ref_set - vllm_set:
        print(f"Missing: {ref_set - vllm_set}")
    if vllm_set - ref_set:
        print(f"Extra:   {vllm_set - ref_set}")
    print(f"Latency: {vllm_latency:.1f}ms")
    print("✅ PARITY OK" if f1 >= 0.8 else "⚠️  NEEDS WORK")


def _decode_entities(logits, words, id_to_classes, word_positions, text, threshold):
    """Decode span logits into entity dicts."""
    probs = logits.sigmoid()
    b_idx, s_idx, k_idx, c_idx = torch.where(probs > threshold)
    if b_idx.numel() == 0:
        return []

    sc = probs[b_idx, s_idx, k_idx, c_idx]
    end_i = s_idx + k_idx
    valid = (end_i + 1) <= len(words)
    b_idx, s_idx, end_i, c_idx, sc = b_idx[valid], s_idx[valid], end_i[valid], c_idx[valid], sc[valid]

    spans = sorted(zip(s_idx.tolist(), end_i.tolist(), c_idx.tolist(), sc.tolist()), key=lambda x: -x[3])
    keep = []
    for cand in spans:
        s, e, c, score = cand
        if not any(not (e < ks or s > ke) for ks, ke, _, _ in keep):
            keep.append(cand)
    keep.sort(key=lambda x: x[0])

    entities = []
    for s, e, c, score in keep:
        if s < len(word_positions) and e < len(word_positions):
            sc_ = word_positions[s][0]
            ec_ = word_positions[e][1]
            entities.append({"text": text[sc_:ec_], "label": id_to_classes[c + 1], "score": round(score, 4), "start": sc_, "end": ec_})
    return entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeBERTa v2 GLiNER Parity Test")
    parser.add_argument("--prepare", action="store_true", help="Phase 1: generate references + model dir")
    parser.add_argument("--test", action="store_true", help="Phase 2: vLLM inference + comparison")
    args = parser.parse_args()

    if args.prepare:
        phase_prepare()
    elif args.test:
        phase_test()
    else:
        # Run both phases in separate processes to avoid CUDA fork issues
        print("Running both phases in separate processes...\n")
        r1 = subprocess.run([sys.executable, __file__, "--prepare"], cwd=os.getcwd())
        if r1.returncode != 0:
            sys.exit(r1.returncode)
        r2 = subprocess.run([sys.executable, __file__, "--test"], cwd=os.getcwd())
        sys.exit(r2.returncode)
