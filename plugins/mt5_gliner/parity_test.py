"""
MT5 GLiNER Parity Test — gliner-x-large (MT5 backbone)

Step 1: Get reference output from GLiNER library
Step 2: Prepare local model dir with proper config.json for vLLM
Step 3: Get vLLM output using mt5_gliner plugin
Step 4: Compare decoded entities and benchmark

Usage:
    pip install gliner stanza langdetect
    VLLM_PLUGINS=mt5_gliner python plugins/mt5_gliner/parity_test.py
"""

import json
import os
import re
import time

import torch

MODEL = "knowledgator/gliner-x-large"
LOCAL_MODEL_DIR = "/tmp/gliner-x-large-vllm"

TEXT = (
    "Cristiano Ronaldo dos Santos Aveiro born 5 February 1985 is a Portuguese "
    "professional footballer who plays as a forward for and captains both Saudi "
    "Pro League club Al Nassr and the Portugal national team. Widely regarded as "
    "one of the greatest players of all time, Ronaldo has won five Ballon d Or "
    "awards, a record three UEFA Men Player of the Year Awards, and four European "
    "Golden Shoes. He has won 33 trophies in his career, including seven league "
    "titles, five UEFA Champions Leagues, the UEFA European Championship and the "
    "UEFA Nations League."
)

LABELS = ["person", "award", "date", "competitions", "teams"]
THRESHOLD = 0.5
MAX_WIDTH = 12
ENT_TOKEN = "<<ENT>>"
SEP_TOKEN = "<<SEP>>"
WORD_PATTERN = re.compile(r'\w+(?:[-_]\w+)*|\S')


def split_text_into_words(text):
    """Split text into (word, start, end) tuples. Matches superpod's preprocessing."""
    return [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(text)]


def run_gliner_reference():
    """Get reference entities from GLiNER library."""
    from gliner import GLiNER

    print("=" * 60)
    print("STEP 1: GLiNER Library Reference")
    print("=" * 60)

    # Large encoder init (trunc_normal_/erfinv) is very slow on CPU; force GPU.
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    model = GLiNER.from_pretrained(MODEL, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Warmup
    _ = model.predict_entities(TEXT[:50], LABELS[:1], threshold=THRESHOLD)

    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        entities = model.predict_entities(TEXT, LABELS, threshold=THRESHOLD)
    ref_latency = (time.perf_counter() - t0) / N * 1000

    print(f"\nEntities ({len(entities)}):")
    for e in entities:
        print(f"  {e['text']:40s} => {e['label']:15s} (score: {e['score']:.4f})")
    print(f"\nLatency: {ref_latency:.1f}ms (avg {N} runs)")

    return entities, ref_latency, model


def prepare_local_model(gliner_model):
    """Create a local model directory with proper config.json for vLLM."""
    print("\n" + "=" * 60)
    print("STEP 2: Preparing Local Model Directory")
    print("=" * 60)

    gliner_config = gliner_model.config.__dict__.copy()
    encoder_config = gliner_config.get("encoder_config", {})
    if hasattr(encoder_config, "__dict__"):
        encoder_config = encoder_config.__dict__

    tokenizer = gliner_model.data_processor.transformer_tokenizer

    ent_token_id = tokenizer.convert_tokens_to_ids(ENT_TOKEN)
    sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

    vllm_config = {
        "model_type": "gliner_mt5",
        "architectures": ["GLiNERMT5Model"],
        # vLLM pooling: num_hidden_layers=0 -> no KV cache allocation
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "hidden_size": encoder_config.get("d_model", 1024),
        # MT5 encoder params
        "vocab_size": encoder_config.get("vocab_size", 250112),
        "d_model": encoder_config.get("d_model", 1024),
        "d_kv": encoder_config.get("d_kv", 64),
        "d_ff": encoder_config.get("d_ff", 2816),
        "num_layers": encoder_config.get("num_layers", 24),
        "num_decoder_layers": encoder_config.get("num_decoder_layers", 24),
        "num_heads": encoder_config.get("num_heads", 16),
        "relative_attention_num_buckets": encoder_config.get("relative_attention_num_buckets", 32),
        "relative_attention_max_distance": encoder_config.get("relative_attention_max_distance", 128),
        "dropout_rate": encoder_config.get("dropout_rate", 0.1),
        "layer_norm_epsilon": encoder_config.get("layer_norm_epsilon", 1e-6),
        "feed_forward_proj": encoder_config.get("feed_forward_proj", "gated-gelu"),
        "is_encoder_decoder": False,
        "use_cache": False,
        "tie_word_embeddings": False,
        # GLiNER params
        "gliner_dropout": gliner_config.get("dropout", 0.3),
        "gliner_hidden_size": gliner_config.get("hidden_size", 768),
        "max_width": gliner_config.get("max_width", 12),
        "class_token_index": ent_token_id,
        "sep_token_index": sep_token_id,
        "has_rnn": gliner_config.get("has_rnn", True),
        "embed_ent_token": gliner_config.get("embed_ent_token", True),
        "max_len": gliner_config.get("max_len", 1024),
    }

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    with open(os.path.join(LOCAL_MODEL_DIR, "config.json"), "w") as f:
        json.dump(vllm_config, f, indent=2)
    print(f"Config saved: d_model={vllm_config['d_model']}, hidden={vllm_config['gliner_hidden_size']}")

    # Save weights with shared tensor dedup
    state_dict = gliner_model.model.state_dict()
    deduped = {}
    seen = set()
    for k, v in state_dict.items():
        ptr = v.data_ptr()
        if ptr not in seen:
            deduped[k] = v.contiguous()
            seen.add(ptr)
        else:
            deduped[k] = v.clone().contiguous()

    import safetensors.torch
    safetensors.torch.save_file(deduped, os.path.join(LOCAL_MODEL_DIR, "model.safetensors"))
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    print(f"Weights: {len(deduped)} tensors, tokenizer saved")

    return vllm_config


def run_vllm_gliner():
    """Get predictions from vLLM mt5_gliner plugin.

    Follows superpod's gliner_preprocessor.py exactly:
    1. Split text into words using regex
    2. Build prompt as [ENT, label1, ENT, label2, ..., SEP, word1, word2, ...]
    3. Tokenize with is_split_into_words=True
    4. Build words_mask from tokenizer.word_ids()
    5. Build span_idx and span_mask
    6. Send via TokensPrompt + PoolingParams(extra_kwargs=...)
    """
    from transformers import AutoTokenizer
    from vllm import LLM, PoolingParams
    from vllm.inputs import TokensPrompt

    print("\n" + "=" * 60)
    print("STEP 3: vLLM mt5_gliner Plugin")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)

    # --- 1. Split text into words (matches superpod's regex) ---
    words_with_pos = split_text_into_words(TEXT)
    words = [w[0] for w in words_with_pos]
    word_positions = [(w[1], w[2]) for w in words_with_pos]
    text_length = len(words)

    print(f"Words: {text_length}")
    print(f"  First 5: {words[:5]}")

    # --- 2. Build prompt as word list (matches superpod) ---
    prompt_list = [token for label in LABELS for token in (ENT_TOKEN, label)]
    prompt_list.append(SEP_TOKEN)
    prompt_len = len(prompt_list)  # number of "words" in prompt
    input_words = prompt_list + words

    # --- 3. Tokenize with is_split_into_words=True ---
    tokenized = tokenizer(
        input_words,
        is_split_into_words=True,
        return_tensors='pt',
        truncation=True,
        padding=False,
    )
    input_ids = tokenized['input_ids'][0]  # (seq_len,)
    attention_mask = tokenized['attention_mask'][0]

    print(f"Input tokens: {len(input_ids)}")

    # --- 4. Build words_mask from word_ids() (matches superpod exactly) ---
    word_ids_list = tokenized.word_ids(batch_index=0)
    word_ids = torch.tensor(
        [w if w is not None else -1 for w in word_ids_list],
        dtype=torch.long
    )
    prev_word_ids = torch.roll(word_ids, 1, dims=0)
    prev_word_ids[0] = -1

    is_new_word = (word_ids != -1) & (word_ids != prev_word_ids)
    is_in_text = (word_ids >= prompt_len)
    valid_indices = is_new_word & is_in_text

    words_mask = torch.zeros_like(word_ids)
    words_mask[valid_indices] = word_ids[valid_indices] - prompt_len + 1

    print(f"Words mask: max={words_mask.max().item()}, non-zero={(words_mask > 0).sum().item()}")

    # --- 5. Build span_idx and span_mask (matches superpod) ---
    K = MAX_WIDTH
    starts = torch.arange(text_length).unsqueeze(1)
    widths = torch.arange(K).unsqueeze(0)
    span_starts = starts.expand(-1, K)
    span_ends = span_starts + widths
    span_idx = torch.stack([span_starts, span_ends], dim=-1).view(-1, 2).unsqueeze(0)
    span_mask = ((span_starts < text_length) & (span_ends < text_length)).view(-1).unsqueeze(0)

    # --- 6. Create extra_kwargs (matches superpod's model_requests.py) ---
    gliner_data = {
        "input_ids": input_ids.tolist(),
        "words_mask": words_mask.tolist(),
        "text_lengths": text_length,
        "attention_mask": attention_mask.tolist(),
        "span_idx": span_idx[0].tolist(),
        "span_mask": span_mask[0].tolist(),
    }

    prompt = TokensPrompt(prompt_token_ids=input_ids.tolist())
    pooling_params = PoolingParams(extra_kwargs=gliner_data)

    # --- 7. Load vLLM and run ---
    vllm_model = LLM(
        model=LOCAL_MODEL_DIR,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
    )

    # Warmup
    _ = vllm_model.embed([prompt], pooling_params=pooling_params)

    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        outputs = vllm_model.embed([prompt], pooling_params=pooling_params)
    vllm_latency = (time.perf_counter() - t0) / N * 1000

    raw = outputs[0].outputs.embedding
    scores = torch.tensor(raw)
    print(f"vLLM output shape: {scores.shape}")
    print(f"Latency: {vllm_latency:.1f}ms (avg {N} runs)")

    # --- 8. Decode entities (matches superpod's postprocessor) ---
    # Extract shape prefix [L, K, C] and reshape
    if scores.dim() == 1 and scores.numel() > 3:
        L = int(scores[0].item())
        K_out = int(scores[1].item())
        C = int(scores[2].item())
        logits = scores[3:].reshape(1, L, K_out, C)  # (B=1, L, K, C)
    else:
        logits = scores.unsqueeze(0) if scores.dim() == 3 else scores

    # Use superpod-style decoding
    id_to_classes = {i + 1: label for i, label in enumerate(LABELS)}
    entities = decode_entities(logits, [words], id_to_classes, word_positions, TEXT, THRESHOLD)

    print(f"\nEntities ({len(entities)}):")
    for e in entities:
        print(f"  {e['text']:40s} => {e['label']:15s} (score: {e['score']:.4f})")

    return entities, scores, vllm_latency


def decode_entities(logits, tokens, id_to_classes, word_positions, text, threshold):
    """Decode entities from GLiNER logits. Matches superpod's GLiNERDecoder + get_final_entities."""
    B, L, K, C = logits.shape
    probs = logits.sigmoid()

    b_idx, s_idx, k_idx, c_idx = torch.where(probs > threshold)
    if b_idx.numel() == 0:
        return []

    scores = probs[b_idx, s_idx, k_idx, c_idx]
    end_inclusive = s_idx + k_idx  # end word index (inclusive)

    # Filter by sequence length
    seq_len = torch.tensor([len(t) for t in tokens])
    end_exclusive = end_inclusive + 1
    valid = end_exclusive <= seq_len[b_idx]

    if not valid.any():
        return []

    b_idx = b_idx[valid]
    s_idx = s_idx[valid]
    end_inclusive = end_inclusive[valid]
    c_idx = c_idx[valid]
    scores = scores[valid]

    # Build span list for NMS
    spans = []
    for s, e, c, sc in zip(s_idx.tolist(), end_inclusive.tolist(), c_idx.tolist(), scores.tolist()):
        label = id_to_classes[c + 1]
        spans.append((s, e, label, sc))

    # Greedy NMS (sort by score desc, remove overlapping)
    spans.sort(key=lambda x: -x[3])
    keep = []
    for cand in spans:
        s, e, lab, score = cand
        clash = False
        for k in keep:
            ks, ke, klab, _ = k
            # Check overlap (non-nested)
            if not (e < ks or s > ke):
                clash = True
                break
        if not clash:
            keep.append(cand)
    keep.sort(key=lambda x: x[0])

    # Convert word indices to character spans
    entities = []
    for s, e, label, score in keep:
        if s < len(word_positions) and e < len(word_positions):
            start_char = word_positions[s][0]
            end_char = word_positions[e][1]
            entity_text = text[start_char:end_char]
            entities.append({
                "text": entity_text,
                "label": label,
                "score": round(score, 4),
                "start": start_char,
                "end": end_char,
            })

    return entities


def compare(ref_entities, vllm_entities, ref_latency, vllm_latency):
    """Compare entity predictions and performance."""
    print("\n" + "=" * 60)
    print("STEP 4: Comparison")
    print("=" * 60)

    ref_set = {(e["text"], e["label"]) for e in ref_entities}
    vllm_set = {(e["text"], e["label"]) for e in vllm_entities}
    matched = ref_set & vllm_set

    precision = len(matched) / len(vllm_set) if vllm_set else 1
    recall = len(matched) / len(ref_set) if ref_set else 1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Reference:  {len(ref_entities)} entities")
    print(f"vLLM:       {len(vllm_entities)} entities")
    print(f"Matched:    {len(matched)}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1:         {f1:.4f}")

    if ref_set - vllm_set:
        print("\nMissing from vLLM:")
        for t, l in ref_set - vllm_set:  # noqa: E741
            print(f"  {t} => {l}")
    if vllm_set - ref_set:
        print("\nExtra in vLLM:")
        for t, l in vllm_set - ref_set:  # noqa: E741
            print(f"  {t} => {l}")

    print(f"\n{'=' * 60}")
    print("Performance")
    print(f"{'=' * 60}")
    print(f"Vanilla GLiNER: {ref_latency:.1f}ms")
    print(f"vLLM GLiNER:    {vllm_latency:.1f}ms")
    speedup = ref_latency / vllm_latency if vllm_latency > 0 else 0
    print(f"Speedup:        {speedup:.2f}x")

    if f1 >= 0.8:
        print(f"\n✅ GOOD PARITY (F1={f1:.4f})")
    else:
        print(f"\n⚠️  PARITY NEEDS WORK (F1={f1:.4f})")


if __name__ == "__main__":
    import argparse  # noqa: E401, I001
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description="MT5 GLiNER Parity Test")
    parser.add_argument("--prepare", action="store_true", help="Phase 1 only: GLiNER reference + model dir")
    parser.add_argument("--test", action="store_true", help="Phase 2 only: vLLM inference + compare")
    args = parser.parse_args()

    REF_FILE = "/tmp/mt5-gliner-reference.json"

    if args.prepare:
        ref_entities, ref_latency, gliner_model = run_gliner_reference()
        prepare_local_model(gliner_model)
        import json as _json
        with open(REF_FILE, "w") as _f:
            _json.dump({"entities": ref_entities, "latency": ref_latency}, _f, indent=2, default=str)
        print(f"Saved references to {REF_FILE}")
    elif args.test:
        import json as _json
        with open(REF_FILE) as _f:
            _ref = _json.load(_f)
        ref_entities = _ref["entities"]
        ref_latency = _ref["latency"]
        vllm_entities, vllm_scores, vllm_latency = run_vllm_gliner()
        compare(ref_entities, vllm_entities, ref_latency, vllm_latency)
    else:
        print("Running both phases in separate processes...\n")
        r1 = subprocess.run([sys.executable, __file__, "--prepare"], cwd=os.getcwd())
        if r1.returncode != 0:
            sys.exit(r1.returncode)
        r2 = subprocess.run([sys.executable, __file__, "--test"], cwd=os.getcwd())
        sys.exit(r2.returncode)
