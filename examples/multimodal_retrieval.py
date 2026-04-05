"""
Example: Multimodal Document Retrieval with vLLM Factory

Sends text queries and image documents to a running ColQwen3 server
for vision-language retrieval (ColPali-style late interaction).

Start the server first:
    vllm serve VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1 \
      --runner pooling --trust-remote-code --dtype bfloat16 \
      --no-enable-prefix-caching --no-enable-chunked-prefill \
      --max-model-len 8192 --limit-mm-per-prompt '{"image": 1}' \
      --io-processor-plugin colqwen3_io --port 8000

Then run:
    python examples/multimodal_retrieval.py
"""

import base64
import math
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
MODEL = "VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1"


def embed_text(text: str, is_query: bool = True) -> list[list[float]]:
    """Encode a text query as multi-vector embedding."""
    resp = requests.post(
        f"{BASE_URL}/pooling",
        json={
            "model": MODEL,
            "data": {"text": text, "is_query": is_query},
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    if data and isinstance(data[0], list):
        return data
    dim = 128
    return [data[i:i + dim] for i in range(0, len(data), dim)]


def embed_image(image_path: str, is_query: bool = False) -> list[list[float]]:
    """Encode an image document as multi-vector embedding."""
    img_bytes = Path(image_path).read_bytes()
    data_uri = f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"

    resp = requests.post(
        f"{BASE_URL}/pooling",
        json={
            "model": MODEL,
            "data": {"image": data_uri, "is_query": is_query},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    if data and isinstance(data[0], list):
        return data
    dim = 128
    return [data[i:i + dim] for i in range(0, len(data), dim)]


def maxsim(query_vecs: list[list[float]], doc_vecs: list[list[float]]) -> float:
    """MaxSim scoring for late interaction."""
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


queries = [
    "What is the main topic of this document?",
    "Summarise the key findings.",
]

print("Encoding text queries...")
for q in queries:
    emb = embed_text(q)
    print(f"  '{q[:50]}' → {len(emb)} token vectors")

print("\nTo encode images, pass local file paths:")
print("  emb = embed_image('path/to/document.png')")
print("  score = maxsim(query_emb, doc_emb)")
