# %% [markdown]
# # GLiNER2 — Schema-Driven Extraction with vLLM Factory
#
# GLiNER2 extends GLiNER with:
# - **Entity extraction** · **Classification** · **Relations** · **JSON schema**
#
# **3.5× throughput** vs vanilla GLiNER2 at batch 8. F1 = 1.0000 parity.

# %%
import os
import subprocess
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "plugins"))
sys.path.insert(0, os.path.join(ROOT, "models"))

# %% [markdown]
# ## Step 1: Prepare Model Directory
#
# GLiNER2 uses a DeBERTa-v3 backbone with plugin-specific config fields.
# The parity test's `--prepare` phase creates the correct model directory.

# %%
MODEL_DIR = "/tmp/gliner2-vllm"

if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print("📦 Preparing GLiNER2 model directory...")
    result = subprocess.run(
        [sys.executable, "plugins/deberta_gliner2/parity_test.py", "--prepare"],
        cwd=ROOT,
        capture_output=True, text=True
    )
    print(result.stdout[-200:] if result.stdout else "")
    if result.returncode != 0:
        print(f"Error: {result.stderr[-300:]}")
        raise RuntimeError("Model preparation failed")
else:
    print(f"✅ Model already prepared at {MODEL_DIR}")

# %% [markdown]
# ## Step 2: Load via vLLM

# %%
from transformers import AutoTokenizer
from vllm import LLM, PoolingParams
from vllm.inputs import TokensPrompt

llm = LLM(
    model=MODEL_DIR,
    trust_remote_code=True,
    enforce_eager=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.5,
    enable_prefix_caching=False,
    disable_log_stats=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("✅ GLiNER2 loaded via vLLM (DeBERTa-v3-large)")

# %% [markdown]
# ## Step 3: Run Entity Extraction

# %%
from deberta_gliner2.processor import (
    decode_entities_from_scores,
    prepare_gliner2_input,
)

text = (
    "Elon Musk, CEO of Tesla and SpaceX, announced a $500 million investment in "
    "a new Gigafactory in Austin, Texas. The facility will produce the Cybertruck "
    "and next-generation battery cells. Construction begins Q1 2025."
)
labels = ["person", "organization", "location", "money", "product", "date"]

# Preprocess: same pipeline as the parity test
prompt_data = prepare_gliner2_input(text, labels, tokenizer)
prompt = TokensPrompt(prompt_token_ids=prompt_data["input_ids"])
pooling_params = PoolingParams(extra_kwargs=prompt_data)

# Run inference
outputs = llm.embed([prompt], pooling_params=pooling_params)
raw = outputs[0].outputs.embedding
scores = torch.tensor(raw)
print(f"✅ Inference complete — output shape: {scores.shape}")

# Decode entities
entities = decode_entities_from_scores(scores, text, labels, threshold=0.5)
print(f"\n🏷️  Extracted {len(entities)} entities:\n")
for ent in entities:
    label = ent.get("label", ent.get("class", "?"))
    entity_text = ent.get("text", ent.get("span", "?"))
    score = ent.get("score", ent.get("confidence", 0))
    print(f"  [{label:>12}]  {entity_text:<30}  (score: {score:.3f})")

# %% [markdown]
# ## Step 4: Throughput Benchmark

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
# GLiNER2 via vLLM Factory: F1 = 1.0000, 3.5× throughput at B=8.
# Full schema extraction (NER, classification, relations, JSON):
# see `plugins/deberta_gliner2/parity_test.py`
