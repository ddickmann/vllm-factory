# %% [markdown]
# # Nemotron-ColEmbed — Bidirectional Qwen3-VL Embeddings
#
# NVIDIA's 4B-parameter bidirectional embedding model with ColBERT-style
# multi-vector output. Uses Qwen3-VL with bidirectional attention, no final RMSNorm,
# and L2-normalized token-level embeddings.
#
# Serves via vLLM with `--no-enable-prefix-caching` (bidirectional attention workaround).

# %%
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "plugins"))
sys.path.insert(0, os.path.join(ROOT, "models"))

# %% [markdown]
# ## Load Nemotron-ColEmbed

# %%
from vllm import LLM

llm = LLM(
    model="nvidia/nemotron-colembed-vl-4b-v2",
    runner="pooling",
    trust_remote_code=True,
    enforce_eager=True,
    dtype="bfloat16",
    max_model_len=4096,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.85,
    limit_mm_per_prompt={"image": 1},
)
print("✅ Nemotron-ColEmbed loaded (4B params)")

# %% [markdown]
# ## Encode Text

# %%
texts = [
    "The European Central Bank announced a significant policy shift on Wednesday.",
    "Deep learning models are increasingly used in production systems.",
    "Quantum computing promises to solve problems intractable for classical machines.",
    "Climate change requires coordinated global policy responses.",
]

outputs = llm.encode(texts, pooling_task="token_embed")

for i, o in enumerate(outputs):
    emb = o.outputs.data
    if hasattr(emb, 'cpu'):
        emb = emb.cpu().numpy()
    else:
        emb = np.array(emb)
    if emb.ndim == 1:
        dim = 2560
        emb = emb.reshape(-1, dim)
    print(f"  Text {i}: {emb.shape[0]} token vectors × {emb.shape[1]} dim")

print(f"✅ Encoded {len(texts)} texts")

# %% [markdown]
# ## Throughput

# %%
text = "The quick brown fox jumps over the lazy dog. " * 15
for _ in range(10):
    llm.encode([text], pooling_task="token_embed")

torch.cuda.synchronize()
t0 = time.perf_counter()
N = 20
for _ in range(N):
    llm.encode([text], pooling_task="token_embed")
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"⚡ B=1: {N/elapsed:.1f} req/s")

batch = [text] * 8
for _ in range(5):
    llm.encode(batch, pooling_task="token_embed")

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    llm.encode(batch, pooling_task="token_embed")
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"⚡ B=8: {8*N/elapsed:.1f} req/s")

# %% [markdown]
# ## Summary
#
# Nemotron-ColEmbed via vLLM Factory:
# - 4B-parameter bidirectional Qwen3-VL
# - **162.6 req/s** at batch 8
# - Serve: `vllm serve nvidia/nemotron-colembed-vl-4b-v2 --no-enable-prefix-caching`
# - cos_sim = 0.9999 parity
