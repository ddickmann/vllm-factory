"""
Example: Embedding Similarity Search with vLLM Factory

Sends queries and documents to a running EmbeddingGemma server
and finds the most similar documents using cosine similarity.

Start the server first:
    vllm serve unsloth/embeddinggemma-300m \
      --runner pooling --trust-remote-code --dtype bfloat16 \
      --no-enable-prefix-caching \
      --io-processor-plugin embeddinggemma_io --port 8000

Then run:
    python examples/embedding_search.py
"""

import requests
import math

BASE_URL = "http://localhost:8000"
MODEL = "unsloth/embeddinggemma-300m"


def embed(text: str) -> list[float]:
    resp = requests.post(
        f"{BASE_URL}/pooling",
        json={"model": MODEL, "data": {"text": text}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


queries = [
    "What is the capital of France?",
    "How do neural networks learn?",
]

documents = [
    "Paris is the capital and largest city of France.",
    "Berlin is the capital of Germany.",
    "Neural networks learn by adjusting weights through backpropagation.",
    "The stock market experienced a downturn last week.",
    "Deep learning models are trained using gradient descent optimization.",
]

print("Encoding queries and documents...\n")
query_embs = [embed(q) for q in queries]
doc_embs = [embed(d) for d in documents]

print(f"{'Query':<45} {'Best Match':<55} {'Score':>6}")
print("-" * 110)

for i, query in enumerate(queries):
    scores = [cosine_sim(query_embs[i], de) for de in doc_embs]
    best_idx = max(range(len(scores)), key=lambda j: scores[j])
    print(f"  {query:<43} {documents[best_idx]:<53} {scores[best_idx]:>6.4f}")

    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    for rank, (doc_idx, score) in enumerate(ranked):
        marker = " <--" if doc_idx == best_idx else ""
        print(f"    #{rank+1} (sim={score:.4f}) {documents[doc_idx][:60]}{marker}")
    print()
