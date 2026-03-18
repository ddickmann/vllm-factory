"""
Example: Entity Linking with vLLM Factory

Sends text + labels + candidate labels to a running GLiNER-Linker server.
The IOProcessor extracts entities and links them to candidates.

Start the server first:
    vllm serve plugins/deberta_gliner_linker/_model_cache \
      --runner pooling --trust-remote-code --dtype bfloat16 \
      --no-enable-prefix-caching --no-enable-chunked-prefill \
      --io-processor-plugin deberta_gliner_linker_io --port 8000

Then run:
    python examples/entity_linking.py
"""

import requests

BASE_URL = "http://localhost:8000"
MODEL = "plugins/deberta_gliner_linker/_model_cache"


def link_entities(
    text: str,
    labels: list[str],
    candidate_labels: list[str],
    threshold: float = 0.3,
) -> list[dict]:
    """Extract entities and link them to candidate labels."""
    resp = requests.post(
        f"{BASE_URL}/pooling",
        json={
            "model": MODEL,
            "data": {
                "text": text,
                "labels": labels,
                "threshold": threshold,
                "candidate_labels": candidate_labels,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"]


text = (
    "Tesla announced record earnings. Elon Musk presented "
    "at the shareholder meeting in Austin, Texas."
)
labels = ["company", "person", "location"]
candidate_labels = [
    "Tesla Inc.",
    "Tesla (physicist)",
    "Elon Musk",
    "Austin, Texas",
    "Austin Powers",
    "Texas Instruments",
]

print(f"Text: {text}\n")
print(f"Entity labels: {labels}")
print(f"Candidate labels: {candidate_labels}\n")

result = link_entities(text, labels, candidate_labels)

print(f"{'Entity':<30} {'Label':<15} {'Linked To':<25} {'Score':>8}")
print("-" * 82)
for entity in result:
    linked = entity.get("linked_to", "—")
    print(f"  {entity['text']:<28} {entity['label']:<15} {linked:<25} {entity['score']:>8.4f}")

if not result:
    print("  (no entities found)")
