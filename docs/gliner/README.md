# Knowledgator GLiNER Plugins (L3 Linker + L4 Rerank)

## Status: production-ready

All GLiNER plugins — including the Knowledgator **L3 linker** (`deberta_gliner_linker`) and **L4 reranker** (`modernbert_gliner_rerank`) — pass end-to-end parity testing via `vllm serve` with IOProcessor plugins.

| Plugin | Parity Score | Test |
|---|---|---|
| `deberta_gliner_linker` (L3) | entity_F1=1.000, link_match=1.000 | `scripts/serve_parity_test.py --plugin deberta_gliner_linker` |
| `modernbert_gliner_rerank` (L4) | entity_F1=0.769 (threshold=0.6) | `scripts/serve_parity_test.py --plugin modernbert_gliner_rerank` |

---

## Serving

Both plugins use IOProcessors for server-side pre/post-processing:

```bash
# L3: Entity Linker
vllm serve plugins/deberta_gliner_linker/_model_cache \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin deberta_gliner_linker_io

# L4: Reranker
vllm serve plugins/modernbert_gliner_rerank/_model_cache \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin modernbert_gliner_rerank_io
```

---

## Documentation index

| Doc | Purpose |
|-----|---------|
| [`OVERVIEW.md`](OVERVIEW.md) | Model table, env vars, links to scripts |
| [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) | Full handover: patches, quirks, usage, verification |
| [`L3_STATUS.md`](L3_STATUS.md) | L3 parity matrix and implementation notes |
| [`L4_NOTES.md`](L4_NOTES.md) | L4 model architecture details |
| [`L4_PARITY.md`](L4_PARITY.md) | L4 parity test status |

## Parity scripts

All parity and integration test scripts live under **[`scripts/gliner/`](../../scripts/gliner/README.md)**.

The main end-to-end test covering all 12 plugins (including L3/L4) is `scripts/serve_parity_test.py`.
