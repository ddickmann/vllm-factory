# %% [markdown]
# # GLiNER — Named Entity Recognition with vLLM Factory
#
# Extract entities from text using GLiNER span-scoring models via vLLM.
# Three backbone options — all achieving **F1 = 1.0000** parity:
# - **ModernBERT** (`mmbert_gliner`) — fast, English-optimized
# - **mT5** (`mt5_gliner`) — multilingual, 1.9× faster than vanilla
# - **DeBERTa** (`deberta_gliner`) — production workhorse
#
# This notebook demonstrates the full pipeline: model prep → inference → entity decode.

# %%
import os
import re
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "plugins"))
sys.path.insert(0, os.path.join(ROOT, "models"))

# %% [markdown]
# ## Step 1: Prepare Model for vLLM
#
# GLiNER models need a local directory with a vLLM-compatible config.json.
# The `forge.model_prep` helper handles this automatically.

# %%
from forge.model_prep import prepare_gliner_model

model_dir = prepare_gliner_model(
    hf_model_id="VAGOsolutions/SauerkrautLM-GLiNER",
    plugin="mmbert_gliner",
)
print(f"Model ready at: {model_dir}")

# %% [markdown]
# ## Step 2: Load via vLLM

# %%
from transformers import AutoTokenizer
from vllm import LLM, PoolingParams
from vllm.inputs import TokensPrompt

llm = LLM(
    model=model_dir,
    trust_remote_code=True,
    enforce_eager=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.5,
    enable_prefix_caching=False,
    disable_log_stats=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
print("✅ ModernBERT-GLiNER loaded via vLLM")

# %% [markdown]
# ## Step 3: Preprocess — Build GLiNER Input

# %%
# Constants
ENT_TOKEN = "<<ENT>>"
SEP_TOKEN = "<<SEP>>"
MAX_WIDTH = 12
WORD_PATTERN = re.compile(r'\w+(?:[-_]\w+)*|\S')

text = (
    "Apple Inc. announced today that CEO Tim Cook will present the new iPhone 16 "
    "at their headquarters in Cupertino, California. The event is scheduled for "
    "September 10, 2024. Goldman Sachs analyst Michael Ng raised the price target "
    "to $250, citing strong demand in the Chinese market."
)
labels = ["Person", "Organization", "Product", "Location", "Date", "Money"]

# 1. Split text into words
words = [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(text)]
word_texts = [w[0] for w in words]
text_length = len(word_texts)

# 2. Build prompt: [ENT, label1, ENT, label2, ..., SEP, word1, word2, ...]
prompt_list = [token for label in labels for token in (ENT_TOKEN, label)]
prompt_list.append(SEP_TOKEN)
prompt_len = len(prompt_list)
input_words = prompt_list + word_texts

# 3. Tokenize
tokenized = tokenizer(input_words, is_split_into_words=True,
                       return_tensors='pt', truncation=True, padding=False)
input_ids = tokenized['input_ids'][0]

# 4. Build words_mask
word_ids_list = tokenized.word_ids(batch_index=0)
word_ids = torch.tensor([w if w is not None else -1 for w in word_ids_list], dtype=torch.long)
prev_word_ids = torch.roll(word_ids, 1, dims=0)
prev_word_ids[0] = -1

is_new_word = (word_ids != -1) & (word_ids != prev_word_ids)
is_in_text = (word_ids >= prompt_len)
valid_indices = is_new_word & is_in_text
words_mask = torch.zeros_like(word_ids)
words_mask[valid_indices] = word_ids[valid_indices] - prompt_len + 1

# 5. Build span_idx and span_mask
starts = torch.arange(text_length).unsqueeze(1)
widths = torch.arange(MAX_WIDTH).unsqueeze(0)
span_starts = starts.expand(-1, MAX_WIDTH)
span_ends = span_starts + widths
span_idx = torch.stack([span_starts, span_ends], dim=-1).view(-1, 2)
span_mask = ((span_starts < text_length) & (span_ends < text_length)).view(-1)

print(f"📝 Text: {text[:80]}...")
print(f"🏷️  Labels: {labels}")
print(f"   {text_length} words → {len(input_ids)} tokens")

# %% [markdown]
# ## Step 4: Run Inference

# %%
gliner_data = {
    "input_ids": input_ids.tolist(),
    "words_mask": words_mask.tolist(),
    "text_lengths": text_length,
    "span_idx": span_idx.tolist(),
    "span_mask": span_mask.tolist(),
}

prompt = TokensPrompt(prompt_token_ids=input_ids.tolist())
pooling_params = PoolingParams(extra_kwargs=gliner_data)

outputs = llm.embed([prompt], pooling_params=pooling_params)
raw = outputs[0].outputs.embedding
scores = torch.tensor(raw)
print(f"✅ Inference complete — output shape: {scores.shape}")

# %% [markdown]
# ## Step 5: Decode Entities

# %%
# Decode: scores format is [L*K, C] or flat with header
if scores.dim() == 1 and scores.numel() > 3:
    L = int(scores[0].item())
    K_out = int(scores[1].item())
    C = int(scores[2].item())
    logits = scores[3:].reshape(1, L, K_out, C)
else:
    logits = scores.unsqueeze(0) if scores.dim() == 2 else scores

# Apply sigmoid for probabilities
probs = torch.sigmoid(logits)

# Extract entities above threshold
THRESHOLD = 0.5
entities = []
if probs.dim() == 4:
    for i in range(probs.shape[1]):
        for j in range(probs.shape[2]):
            for c in range(probs.shape[3]):
                if probs[0, i, j, c].item() > THRESHOLD:
                    start_word = i
                    end_word = min(i + j, text_length - 1)
                    entity_text = " ".join(word_texts[start_word:end_word + 1])
                    entities.append({
                        "text": entity_text,
                        "label": labels[c] if c < len(labels) else f"class_{c}",
                        "score": probs[0, i, j, c].item(),
                    })

# Sort by score
entities.sort(key=lambda x: x["score"], reverse=True)

print(f"\n🏷️  Extracted {len(entities)} entities:\n")
for ent in entities:
    print(f"  [{ent['label']:>12}]  {ent['text']:<30}  (score: {ent['score']:.3f})")

# %% [markdown]
# ## Step 6: Throughput Benchmark

# %%
# Warmup
for _ in range(10):
    llm.embed([prompt], pooling_params=pooling_params)

torch.cuda.synchronize()
t0 = time.perf_counter()
N = 50
for _ in range(N):
    llm.embed([prompt], pooling_params=pooling_params)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"⚡ Throughput (B=1): {N/elapsed:.1f} req/s ({elapsed/N*1000:.1f}ms/req)")

# %% [markdown]
# ## Summary
#
# GLiNER via vLLM Factory:
# - **F1 = 1.0000** entity extraction parity across all 3 backbones
# - Full pipeline: `model_prep → vLLM encode → span decode`
# - Swap backbones: change model + plugin name → same API
# - Production-ready: `vllm serve` for concurrent NER
#
# Other backbones:
# ```python
# # mT5 multilingual
# model_dir = prepare_gliner_model("knowledgator/gliner-multitask-v1.0", plugin="mt5_gliner")
#
# # DeBERTa
# model_dir = prepare_gliner_model("knowledgator/gliner-bi-large-v2.0", plugin="deberta_gliner")
# ```
