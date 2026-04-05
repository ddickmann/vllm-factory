"""
Example: GLiNER Named Entity Recognition with vLLM Factory

Sends text + labels to a running GLiNER server and gets back
structured entities. All pre/post-processing happens server-side
via the IOProcessor — no client-side tokenization needed.

Start the server first:
    # Prepare model (one-time)
    vllm-factory-prep --model VAGOsolutions/SauerkrautLM-GLiNER \
      --output /tmp/sauerkraut-gliner-vllm

    # Serve
    vllm serve /tmp/sauerkraut-gliner-vllm \
      --runner pooling --trust-remote-code --dtype bfloat16 \
      --no-enable-prefix-caching --no-enable-chunked-prefill \
      --io-processor-plugin mmbert_gliner_io --port 8000

Then run:
    python examples/gliner_ner.py
"""


import requests

BASE_URL = "http://localhost:8000"
MODEL = "/tmp/sauerkraut-gliner-vllm"


def extract_entities(text: str, labels: list[str], threshold: float = 0.3) -> list[dict]:
    """Extract named entities from text using the GLiNER IOProcessor."""
    resp = requests.post(
        f"{BASE_URL}/pooling",
        json={
            "model": MODEL,
            "data": {
                "text": text,
                "labels": labels,
                "threshold": threshold,
                "flat_ner": True,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"]


text = (
    "Apple Inc. announced a partnership with OpenAI to integrate ChatGPT "
    "into iOS 18. Tim Cook presented the news at WWDC 2024 in Cupertino, "
    "California. The deal is reportedly worth $500 million."
)
labels = ["company", "person", "product", "location", "money", "event"]

print(f"Text: {text}\n")
print(f"Labels: {labels}\n")

entities = extract_entities(text, labels)

print(f"{'Entity':<40} {'Label':<15} {'Score':>8}")
print("-" * 68)
for entity in entities:
    print(f"  {entity['text']:<38} {entity['label']:<15} {entity['score']:>8.4f}")

if not entities:
    print("  (no entities found)")

print(f"\nTotal: {len(entities)} entities")
