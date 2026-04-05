"""Ad-hoc experiment: entity-specific vs generic labels (GLiNER linker).

Ad-hoc script — not a supported test harness. See docs/gliner/README.md.
Init LLM first, then load labels encoder to avoid fork memory issues."""
import json
import os
import re

import torch

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from plugins.deberta_gliner_linker import get_model_path

model_path = get_model_path()

# Step 1: Init vLLM FIRST (before loading labels encoder into memory)
from vllm import LLM
from vllm.inputs import TokensPrompt
from vllm.pooling_params import PoolingParams

print("Initializing LLM...", flush=True)
llm = LLM(
    model=model_path, trust_remote_code=True, dtype="float32",
    max_model_len=512, enforce_eager=True, enable_prefix_caching=False,
    gpu_memory_utilization=0.5,
)
print("LLM initialized!", flush=True)

# Step 2: Now load labels encoder
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, DebertaConfig, DebertaModel

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

HF_MODEL = "knowledgator/gliner-linker-large-v1.0"
cfg_path = hf_hub_download(HF_MODEL, "gliner_config.json")
with open(cfg_path) as f:
    gliner_cfg = json.load(f)
le_cfg = gliner_cfg["labels_encoder_config"]
labels_enc = DebertaModel(DebertaConfig(**le_cfg))
weights_path = hf_hub_download(HF_MODEL, "pytorch_model.bin")
state = torch.load(weights_path, map_location="cpu", weights_only=True)
prefix = "token_rep_layer.labels_encoder.model."
labels_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
labels_enc.load_state_dict(labels_state, strict=False)
labels_enc.eval()
print("Labels encoder loaded", flush=True)

def encode_labels(labels):
    embs = []
    for l in labels:
        enc = tokenizer(l, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = labels_enc(input_ids=enc["input_ids"])
        hs = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).expand(hs.size()).float()
        mean = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embs.append(mean.squeeze(0))
    return torch.stack(embs, dim=0)

# Tokenize text
text = "Apple announced new products in California. Michael Jordan joined the team."
word_pattern = re.compile(r'\w+(?:[-_]\w+)*|\S')
words = [m.group() for m in word_pattern.finditer(text)]
tok_result = tokenizer([words], is_split_into_words=True, return_tensors="pt",
                       truncation=True, padding="longest")
input_ids = tok_result["input_ids"][0]
word_ids_list = tok_result.word_ids(batch_index=0)
words_mask = torch.zeros(len(word_ids_list), dtype=torch.long)
prev_wid = -1
for idx, wid in enumerate(word_ids_list):
    if wid is not None and wid != prev_wid:
        words_mask[idx] = wid + 1
        prev_wid = wid

def run_and_decode(labels, label_embs, threshold=0.3):
    gliner_data = {
        "input_ids": input_ids.tolist(),
        "words_mask": words_mask.tolist(),
        "text_lengths": len(words),
        "labels_embeds": label_embs.tolist(),
    }
    prompt = TokensPrompt(prompt_token_ids=input_ids.tolist())
    pp = PoolingParams(extra_kwargs=gliner_data)
    outputs = llm.embed([prompt], pooling_params=pp)
    raw = torch.tensor(outputs[0].outputs.embedding)
    W, C, S = int(raw[0]), int(raw[1]), int(raw[2])
    scores = raw[3:].reshape(W, C, S)
    probs = torch.sigmoid(scores)

    print(f"  Scores shape: ({W}, {C}, {S})")
    for c, label in enumerate(labels):
        sp = probs[:, c, 0]
        max_p = sp.max().item()
        max_idx = sp.argmax().item()
        max_w = words[max_idx] if max_idx < len(words) else "?"
        print(f"  {label}: max_start_prob={max_p:.4f} (word='{max_w}')")

    entities = []
    for c in range(C):
        starts = (probs[:, c, 0] > threshold).nonzero(as_tuple=False).flatten()
        ends = (probs[:, c, 1] > threshold).nonzero(as_tuple=False).flatten()
        for st in starts:
            for ed in ends:
                if ed < st:
                    continue
                ins = probs[st:ed+1, c, 2]
                if (ins < threshold).any():
                    continue
                combined = torch.cat([ins, probs[st, c, 0:1], probs[ed, c, 1:2]])
                score = combined.min().item()
                entity_text = " ".join(words[st:ed+1])
                entities.append({"text": entity_text, "label": labels[c], "score": score})
    return entities

# Test 1: Generic type labels
print("\n=== GENERIC TYPE LABELS ===", flush=True)
generic_labels = ["company", "location", "person"]
generic_embs = encode_labels(generic_labels)
ents = run_and_decode(generic_labels, generic_embs, threshold=0.3)
for e in sorted(ents, key=lambda x: -x["score"]):
    print(f"  '{e['text']}' -> {e['label']} (score={e['score']:.4f})")
if not ents:
    print("  (none found)")

# Test 2: Entity-specific labels (like GLinkerBackend uses)
print("\n=== ENTITY-SPECIFIC LABELS ===", flush=True)
entity_labels = [
    "Apple: Technology company",
    "California: US state",
    "Michael Jordan: Basketball player",
    "Google: Search engine",
]
entity_embs = encode_labels(entity_labels)
ents = run_and_decode(entity_labels, entity_embs, threshold=0.3)
for e in sorted(ents, key=lambda x: -x["score"]):
    print(f"  '{e['text']}' -> {e['label']} (score={e['score']:.4f})")
if not ents:
    print("  (none found)")

print("\nDone.", flush=True)
