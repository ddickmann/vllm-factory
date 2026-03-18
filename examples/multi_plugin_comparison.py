"""
Example: Multi-Plugin Comparison

Demonstrates running the same NER query against different GLiNER backends
to compare their entity extraction results.

This requires running each model on a different port. Start two servers:

    # Server 1: ModernBERT GLiNER
    vllm-factory-prep --model VAGOsolutions/SauerkrautLM-GLiNER \
      --output /tmp/sauerkraut-gliner-vllm
    vllm serve /tmp/sauerkraut-gliner-vllm \
      --runner pooling --trust-remote-code --dtype bfloat16 \
      --no-enable-prefix-caching --no-enable-chunked-prefill \
      --io-processor-plugin mmbert_gliner_io --port 8001

    # Server 2: mT5 GLiNER (multilingual)
    vllm-factory-prep --model knowledgator/gliner-x-large \
      --output /tmp/gliner-x-large-vllm
    vllm serve /tmp/gliner-x-large-vllm \
      --runner pooling --trust-remote-code --dtype bfloat16 \
      --no-enable-prefix-caching --no-enable-chunked-prefill \
      --io-processor-plugin mt5_gliner_io --port 8002

Then run:
    python examples/multi_plugin_comparison.py
"""

import requests

TEXT = (
    "The European Central Bank raised interest rates by 25 basis points. "
    "Christine Lagarde announced the decision in Frankfurt on Thursday."
)
LABELS = ["organization", "person", "location", "financial_metric"]

MODELS = [
    {
        "name": "ModernBERT GLiNER (mmbert_gliner)",
        "url": "http://localhost:8001",
        "model": "/tmp/sauerkraut-gliner-vllm",
    },
    {
        "name": "mT5 GLiNER (mt5_gliner)",
        "url": "http://localhost:8002",
        "model": "/tmp/gliner-x-large-vllm",
    },
]


def extract(url: str, model: str, text: str, labels: list[str]) -> list[dict]:
    resp = requests.post(
        f"{url}/pooling",
        json={
            "model": model,
            "data": {"text": text, "labels": labels, "threshold": 0.3, "flat_ner": True},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"]


print(f"Text: {TEXT}\n")
print(f"Labels: {LABELS}\n")

for m in MODELS:
    print(f"{'=' * 60}")
    print(f"  {m['name']}")
    print(f"{'=' * 60}")
    try:
        entities = extract(m["url"], m["model"], TEXT, LABELS)
        for e in entities:
            print(f"  {e['text']:<40} {e['label']:<20} {e['score']:.4f}")
        if not entities:
            print("  (no entities found)")
    except requests.ConnectionError:
        print(f"  Server not running at {m['url']}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
