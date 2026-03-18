"""
ModernColBERT Reference Generator

Generates ground-truth embeddings using PyLate (the reference ColBERT library)
for parity validation against the vLLM plugin.

Usage:
    python plugins/moderncolbert/generate_reference.py \
        --model VAGOsolutions/SauerkrautLM-Multi-ModernColBERT \
        --output-dir /tmp/moderncolbert_reference
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# Representative retrieval queries
QUERIES = [
    "What are the main causes of climate change?",
    "Explain the mechanism of transformer attention in neural networks.",
    "How does the immune system fight viral infections?",
    "What are the key principles of object-oriented programming?",
    "Describe the process of photosynthesis in plants.",
]

# Representative documents (varied lengths)
DOCUMENTS = [
    (
        "Climate change is primarily caused by human activities that release greenhouse gases "
        "such as carbon dioxide and methane into the atmosphere. The burning of fossil fuels "
        "for energy, deforestation, industrial processes, and agricultural practices all "
        "contribute significantly. These gases trap heat from the sun, leading to a "
        "gradual warming of the Earth's surface known as the greenhouse effect."
    ),
    (
        "Transformer attention mechanisms allow neural networks to weigh the importance of "
        "different parts of the input sequence when producing an output. The self-attention "
        "operation computes query, key, and value projections for each token. Attention scores "
        "are computed as scaled dot products between queries and keys, then normalized with "
        "softmax. The resulting weights are used to compute a weighted sum of the values, "
        "enabling the model to focus on relevant context regardless of distance in the sequence."
    ),
    (
        "The immune system fights viral infections through both innate and adaptive responses. "
        "Innate immunity provides rapid, non-specific defense via interferon signaling and "
        "natural killer cells. Adaptive immunity generates virus-specific T-cells and "
        "antibodies through clonal selection of B and T lymphocytes. Memory cells formed "
        "during the primary response enable faster and stronger reactions upon re-exposure "
        "to the same pathogen."
    ),
    (
        "Object-oriented programming (OOP) is organized around four key principles. "
        "Encapsulation bundles data and methods that operate on it within objects, hiding "
        "internal state. Inheritance allows new classes to derive properties from existing "
        "ones, promoting code reuse. Polymorphism enables objects of different types to be "
        "treated through a common interface, dispatching methods dynamically. Abstraction "
        "hides implementation complexity behind well-defined interfaces."
    ),
    (
        "Photosynthesis is the process by which green plants convert light energy into "
        "chemical energy stored as glucose. In the light-dependent reactions occurring in "
        "the thylakoid membranes, chlorophyll absorbs sunlight to split water molecules, "
        "releasing oxygen and generating ATP and NADPH. The light-independent Calvin cycle "
        "in the stroma uses these energy carriers to fix carbon dioxide from the air into "
        "organic sugar molecules through a series of enzymatic reactions."
    ),
]


def time_encode(model, texts: list, is_query: bool, n_warmup: int = 2, n_timed: int = 10) -> dict:
    """Measure per-request latency for PyLate encode."""
    # Warmup
    for _ in range(n_warmup):
        model.encode(texts[:1], is_query=is_query, batch_size=1, show_progress_bar=False)

    latencies_ms = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        model.encode(texts[:1], is_query=is_query, batch_size=1, show_progress_bar=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "req_per_s": round(1000 / (sum(latencies_ms) / n), 2),
        "p50_ms": round(latencies_ms[n // 2], 2),
        "p99_ms": round(latencies_ms[int(n * 0.99)], 2),
    }


def main():
    p = argparse.ArgumentParser(description="ModernColBERT PyLate reference generator")
    p.add_argument("--model", default="VAGOsolutions/SauerkrautLM-Multi-ModernColBERT")
    p.add_argument("--output-dir", default="/tmp/moderncolbert_reference")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("ModernColBERT Reference Generator")
    print(f"  model:  {args.model}")
    print(f"  output: {out}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Load PyLate model
    # ------------------------------------------------------------------ #
    print("\n[1/5] Loading PyLate ColBERT model...")
    from pylate import models
    model = models.ColBERT(model_name_or_path=args.model)
    print("  ✓ Model loaded")

    # ------------------------------------------------------------------ #
    # 2. Encode queries
    # ------------------------------------------------------------------ #
    print("\n[2/5] Computing query embeddings...")
    query_embeddings = model.encode(
        QUERIES,
        is_query=True,
        batch_size=1,
        show_progress_bar=True,
    )
    # pylate returns a list of numpy arrays; convert to tensors
    query_tensors = [torch.as_tensor(e).float() for e in query_embeddings]
    for i, t in enumerate(query_tensors):
        print(f"  query_{i} -> shape {tuple(t.shape)}")
    torch.save(query_tensors, out / "query_embeddings.pt")
    print(f"  ✓ Saved query_embeddings.pt ({len(query_tensors)} tensors)")

    # ------------------------------------------------------------------ #
    # 3. Encode documents
    # ------------------------------------------------------------------ #
    print("\n[3/5] Computing document embeddings...")
    doc_embeddings = model.encode(
        DOCUMENTS,
        is_query=False,
        batch_size=1,
        show_progress_bar=True,
    )
    doc_tensors = [torch.as_tensor(e).float() for e in doc_embeddings]
    for i, t in enumerate(doc_tensors):
        print(f"  doc_{i} -> shape {tuple(t.shape)}")
    torch.save(doc_tensors, out / "document_embeddings.pt")
    print(f"  ✓ Saved document_embeddings.pt ({len(doc_tensors)} tensors)")

    # ------------------------------------------------------------------ #
    # 4. Save texts
    # ------------------------------------------------------------------ #
    (out / "queries.json").write_text(json.dumps(QUERIES, indent=2))
    (out / "documents.json").write_text(json.dumps(DOCUMENTS, indent=2))
    print("\n[4/5] Saved queries.json + documents.json")

    # ------------------------------------------------------------------ #
    # 5. HF baseline throughput
    # ------------------------------------------------------------------ #
    print("\n[5/5] Measuring PyLate baseline throughput...")
    q_stats = time_encode(model, QUERIES, is_query=True)
    d_stats = time_encode(model, DOCUMENTS, is_query=False)
    print(f"  queries:   {q_stats['req_per_s']} req/s  p50={q_stats['p50_ms']}ms  p99={q_stats['p99_ms']}ms")
    print(f"  documents: {d_stats['req_per_s']} req/s  p50={d_stats['p50_ms']}ms  p99={d_stats['p99_ms']}ms")

    stats = {"queries": q_stats, "documents": d_stats}
    (out / "hf_benchmark.json").write_text(json.dumps(stats, indent=2))

    print()
    print("=" * 70)
    print(f"OK Reference outputs saved to {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
