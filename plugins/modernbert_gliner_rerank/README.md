# `modernbert_gliner_rerank` — Knowledgator GLiNER L4 (uni-encoder rerank)

> **Production-ready.** Uses a custom ModernBERT encoder for numerical parity with vanilla GLiNER. All entities validated end-to-end via recall-gated parity testing in bfloat16.

**Model:** [knowledgator/gliner-linker-rerank-v1.0](https://huggingface.co/knowledgator/gliner-linker-rerank-v1.0)  
**Architecture:** ModernBERT (ettin) + projection + bidirectional LSTM + scorer; labels live in the **same** sequence as text (uni-encoder).

## Docs

- **[`docs/gliner/OVERVIEW.md`](../../docs/gliner/OVERVIEW.md)**  
- **[`docs/gliner/INTEGRATION_GUIDE.md`](../../docs/gliner/INTEGRATION_GUIDE.md)**  
- **[`docs/gliner/L4_PARITY.md`](../../docs/gliner/L4_PARITY.md)** — preprocess vs GPU status  

## Parity scripts (`PYTHONPATH=.` from repo root)

| Script | Purpose |
|--------|---------|
| [`scripts/gliner/l4/preprocess_parity_test.py`](../../scripts/gliner/l4/preprocess_parity_test.py) | CPU: collator vs `_tokenize` |
| [`scripts/gliner/l4/entity_parity_test.py`](../../scripts/gliner/l4/entity_parity_test.py) | GPU: native vs processor (often fails) |
| [`scripts/gliner/l4/batch_vllm_parity_test.py`](../../scripts/gliner/l4/batch_vllm_parity_test.py) | GPU: batched vs sequential embed |

**CI-style CPU checks:** `pytest tests/test_modernbert_gliner_rerank_prepare.py tests/test_modernbert_gliner_rerank_batch_contract.py`

## Usage (processor)

```python
from plugins.modernbert_gliner_rerank.processor import GLiNERRerankProcessor

proc = GLiNERRerankProcessor()
proc.warmup(labels)
entities = proc.predict_entities(text, threshold=0.5)
proc.close()
```

## Serve

```bash
export VLLM_PLUGINS=modernbert_gliner_rerank
vllm serve "$(python -c 'from plugins.modernbert_gliner_rerank import get_model_path; print(get_model_path())')" \
  --trust-remote-code --runner pooling
```
