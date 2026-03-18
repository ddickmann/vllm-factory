"""
Example: ColBERT Multi-Vector Retrieval with vLLM Factory

Sends queries and documents to a running ModernColBERT server
and computes MaxSim late-interaction scores.

Start the server first:
    vllm serve VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT \
      --runner pooling --trust-remote-code --dtype bfloat16 \
      --no-enable-prefix-caching --no-enable-chunked-prefill \
      --io-processor-plugin moderncolbert_io --port 8000

Then run:
    python examples/colbert_retrieval.py
"""

import requests

BASE_URL = "http://localhost:8000"
MODEL = "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT"


def encode(text: str) -> list[list[float]]:
    """Get multi-vector embedding (list of token-level vectors)."""
    resp = requests.post(
        f"{BASE_URL}/pooling",
        json={"model": MODEL, "data": {"text": text}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"]


def maxsim(query_vecs: list[list[float]], doc_vecs: list[list[float]]) -> float:
    """ColBERT MaxSim scoring: for each query token, find max cosine to any doc token."""
    import math

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def norm(a):
        return math.sqrt(sum(x * x for x in a))

    total = 0.0
    for qv in query_vecs:
        qn = norm(qv)
        best = -1.0
        for dv in doc_vecs:
            dn = norm(dv)
            sim = dot(qv, dv) / (qn * dn + 1e-9)
            best = max(best, sim)
        total += best
    return total / len(query_vecs)


query = "European Central Bank monetary policy"
documents = [
    "The ECB sets interest rates for the eurozone to maintain price stability.",
    "Machine learning is transforming the financial industry.",
    "Frankfurt is home to the European Central Bank headquarters.",
    "Quantum computing may revolutionize cryptography.",
]

print(f"Query: {query}\n")
print("Encoding query and documents...")

query_emb = encode(query)
doc_embs = [encode(d) for d in documents]

print(f"  Query: {len(query_emb)} token vectors")
for i, de in enumerate(doc_embs):
    print(f"  Doc {i}: {len(de)} token vectors")

print(f"\n{'Rank':<6} {'MaxSim':>8}   {'Document'}")
print("-" * 80)

scores = [(i, maxsim(query_emb, de)) for i, de in enumerate(doc_embs)]
for rank, (i, score) in enumerate(sorted(scores, key=lambda x: -x[1])):
    print(f"  #{rank+1:<4} {score:>8.4f}   {documents[i]}")
