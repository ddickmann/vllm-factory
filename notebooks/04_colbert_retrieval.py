# %% [markdown]
# # ColBERT — Multi-Vector Retrieval with vLLM Factory
#
# ColBERT retrieval using late interaction (MaxSim). Two backbones:
# - **ModernColBERT** — 285 req/s offline, 2,395 req/s server, 16.9× vs PyLate
# - **LFM2-ColBERT** — Mamba/SSM hybrid, 895 req/s at batch 32
#
# Both produce per-token embeddings for fine-grained matching.

# %%
import sys
import os
import time
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "plugins"))
sys.path.insert(0, os.path.join(ROOT, "models"))

# %% [markdown]
# ## Load ModernColBERT

# %%
import moderncolbert  # noqa — registers plugin
from vllm import LLM

llm = LLM(
    model="VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT",
    trust_remote_code=True,
    enforce_eager=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.5,
    disable_log_stats=True,
)
print("✅ ModernColBERT loaded")

# %% [markdown]
# ## Index Documents with Multi-Vector Embeddings

# %%
documents = [
    "Machine learning is a subset of artificial intelligence focused on data-driven learning.",
    "Deep learning uses multi-layer neural networks for complex pattern recognition.",
    "Transformers revolutionized NLP with self-attention mechanisms.",
    "ColBERT uses late interaction for efficient and effective retrieval.",
    "vLLM provides high-throughput serving for large language models.",
    "Retrieval-augmented generation combines search with language models.",
    "Convolutional neural networks excel at image classification tasks.",
    "BERT introduced bidirectional pre-training for language understanding.",
    "GPT models generate text autoregressively from left to right.",
    "Sentence-BERT produces fixed-size sentence embeddings for similarity search.",
    "The European Union proposed new regulations for artificial intelligence systems.",
    "Quantum computing promises exponential speedups for certain computational problems.",
    "Climate models predict significant temperature increases by 2050.",
    "CRISPR gene editing technology enables precise modifications to DNA sequences.",
    "Blockchain provides distributed consensus for decentralized applications.",
    "Docker containers package applications with their dependencies for consistent deployment.",
    "Kubernetes orchestrates containerized workloads across clusters of machines.",
    "PostgreSQL is an advanced open-source relational database management system.",
    "Redis provides in-memory data structures for caching and real-time analytics.",
    "GraphQL offers a flexible query language for APIs as an alternative to REST.",
]

print(f"📚 Indexing {len(documents)} documents...")
t0 = time.perf_counter()
outputs = llm.encode(documents, pooling_task="token_embed")
elapsed = time.perf_counter() - t0

# Extract multi-vector embeddings
doc_embeddings = []
for o in outputs:
    emb = o.outputs.data
    if hasattr(emb, 'cpu'):
        emb = emb.cpu().numpy()
    else:
        emb = np.array(emb)
    if emb.ndim == 1:
        dim = 128  # ColBERT dim
        emb = emb.reshape(-1, dim)
    doc_embeddings.append(emb)

print(f"✅ Indexed in {elapsed:.2f}s ({len(documents)/elapsed:.0f} docs/s)")
for i in range(3):
    print(f"   Doc {i}: {doc_embeddings[i].shape[0]} token vectors × {doc_embeddings[i].shape[1]} dim")

# %% [markdown]
# ## MaxSim Retrieval

# %%
def maxsim_score(query_emb, doc_emb):
    """ColBERT MaxSim: max similarity per query token, then sum."""
    sim = query_emb @ doc_emb.T
    return sim.max(axis=1).sum()


def search(query: str, top_k: int = 5):
    """ColBERT late-interaction search."""
    q_out = llm.encode([query], pooling_task="token_embed")
    q_emb = q_out[0].outputs.data
    if hasattr(q_emb, 'cpu'):
        q_emb = q_emb.cpu().numpy()
    else:
        q_emb = np.array(q_emb)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(-1, 128)

    scores = [(i, maxsim_score(q_emb, d)) for i, d in enumerate(doc_embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🔍 Query: '{query}'")
    for rank, (idx, score) in enumerate(scores[:top_k], 1):
        print(f"   {rank}. [{score:7.2f}] {documents[idx][:80]}...")


search("How does ColBERT retrieval work?")
search("What databases are available?")
search("Tell me about containerization and deployment")
search("Artificial intelligence regulation")

# %% [markdown]
# ## Throughput Benchmark

# %%
texts = ["The quick brown fox jumps over the lazy dog. " * 20] * 32

for bs in [1, 8, 32]:
    batch = texts[:bs]
    for _ in range(5):
        llm.encode(batch, pooling_task="token_embed")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        llm.encode(batch, pooling_task="token_embed")
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 20
    print(f"Batch {bs:3d}: {bs/elapsed:.1f} req/s  ({elapsed*1000:.1f}ms)")

# %% [markdown]
# ## Summary
#
# ColBERT via vLLM Factory:
# - **16.9× faster** than PyLate at batch 1
# - **2,395 req/s** in server mode with continuous batching
# - MaxSim late interaction preserves fine-grained token matching
# - Swap `moderncolbert` → `lfm2_colbert` for Mamba/SSM hybrid (895 req/s at B=32)
