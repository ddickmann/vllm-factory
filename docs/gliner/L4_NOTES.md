# L4 Reranker — `gliner-linker-rerank-v1.0`

> **Production-ready.** See [`L4_PARITY.md`](L4_PARITY.md) and [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md).

Per the [model card](https://huggingface.co/knowledgator/gliner-linker-large-v1.0), **L4** uses [`knowledgator/gliner-linker-rerank-v1.0`](https://huggingface.co/knowledgator/gliner-linker-rerank-v1.0) with **ettin-encoder-68m** (ModernBERT), not DeBERTa. It is **not** the same class as the L3 BiEncoderToken linker.

## Implementation in vLLM Factory

- **Plugin**: [`plugins/modernbert_gliner_rerank`](../../plugins/modernbert_gliner_rerank/) — entry point `modernbert_gliner_rerank`.
- **Handover (L3 + L4)**: [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md).
- **Short overview**: [`OVERVIEW.md`](OVERVIEW.md).

## Card usage snippet (GLinker reference)

```python
builder.l4.configure(
    model="knowledgator/gliner-linker-rerank-v1.0",
    threshold=0.3,
    max_labels=5,
)
```
