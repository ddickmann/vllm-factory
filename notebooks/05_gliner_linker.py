# %% [markdown]
# # GLiNER-Linker — Entity Linking with vLLM Factory
#
# Dual-encoder entity linking: DeBERTa text + label encoders, LSTM + scorer.
# **10.9× throughput** vs vanilla HF pipeline. cos_sim = 1.000000 parity.

# %%
import sys
import os
import time
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "plugins"))
sys.path.insert(0, os.path.join(ROOT, "models"))

# %% [markdown]
# ## Prepare Model

# %%
from forge.model_prep import prepare_gliner_model

model_dir = prepare_gliner_model(
    hf_model_id="knowledgator/gliner-linker-large-v1.0",
    plugin="deberta_gliner_linker",
)

# %% [markdown]
# ## Load via vLLM

# %%
from vllm import LLM

llm = LLM(
    model=model_dir,
    trust_remote_code=True,
    enforce_eager=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.5,
    enable_prefix_caching=False,
    disable_log_stats=True,
)
print("✅ GLiNER-Linker loaded")

# %% [markdown]
# ## Encode Text for Linking

# %%
texts = [
    "Apple is developing new artificial intelligence chips for the iPhone.",
    "Tesla announced a billion dollar investment in autonomous driving technology.",
    "The European Central Bank raised interest rates to fight inflation.",
    "CRISPR gene editing could cure sickle cell disease within a decade.",
    "SpaceX launched another batch of Starlink satellites into orbit.",
]

outputs = llm.embed(texts)

for i, o in enumerate(outputs):
    emb = o.outputs.embedding
    if isinstance(emb, list):
        emb = np.array(emb)
    print(f"  Text {i}: shape={emb.shape if hasattr(emb, 'shape') else len(emb)}")

print(f"\n✅ Encoded {len(texts)} texts for entity linking")

# %% [markdown]
# ## Throughput

# %%
bench_text = "Apple CEO Tim Cook presented the new iPhone at their Cupertino headquarters."

for _ in range(10):
    llm.embed([bench_text])

torch.cuda.synchronize()
t0 = time.perf_counter()
N = 50
for _ in range(N):
    llm.embed([bench_text])
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"⚡ Throughput (B=1): {N/elapsed:.1f} req/s")

batch = [bench_text] * 8
for _ in range(10):
    llm.embed(batch)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    llm.embed(batch)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"⚡ Throughput (B=8): {8*N/elapsed:.1f} req/s")

# %% [markdown]
# ## Summary
#
# GLiNER-Linker: 10.9× throughput, cos_sim = 1.000000 exact parity.
# Full linking pipeline: see `scripts/gliner/l3/parity_test.py` (docs/gliner/README.md).
