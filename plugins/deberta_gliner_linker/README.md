# `deberta_gliner_linker` — Knowledgator GLiNER L3 (entity linking)

> **Production-ready.** Passes end-to-end recall-gated parity testing in bfloat16. See **[`docs/gliner/README.md`](../../docs/gliner/README.md)**.

**Model:** [knowledgator/gliner-linker-large-v1.0](https://huggingface.co/knowledgator/gliner-linker-large-v1.0)  
**Architecture:** Dual DeBERTa v1 (text + label encoders), scorer head (no LSTM on the GLiNER forward path for this checkpoint).

## Docs

- **[`docs/gliner/OVERVIEW.md`](../../docs/gliner/OVERVIEW.md)** — quick integration overview  
- **[`docs/gliner/INTEGRATION_GUIDE.md`](../../docs/gliner/INTEGRATION_GUIDE.md)** — patches, `extra_kwargs`, processors  
- **[`docs/gliner/L3_STATUS.md`](../../docs/gliner/L3_STATUS.md)** — parity matrix  

## Parity scripts (repo root, `PYTHONPATH=.`)

| Script | Purpose |
|--------|---------|
| [`scripts/gliner/l3/parity_test.py`](../../scripts/gliner/l3/parity_test.py) | Logits / cosine parity |
| [`scripts/gliner/l3/preprocess_parity_test.py`](../../scripts/gliner/l3/preprocess_parity_test.py) | Collator vs processor tensors |
| [`scripts/gliner/l3/entity_parity_test.py`](../../scripts/gliner/l3/entity_parity_test.py) | End-to-end entities vs native |

## Usage (processor)

```python
from plugins.deberta_gliner_linker.processor import GLiNERLinkerProcessor

proc = GLiNERLinkerProcessor()
proc.warmup(labels)
entities = proc.predict_entities(text, threshold=0.5)
proc.close()
```

## Serve

```bash
export VLLM_PLUGINS=deberta_gliner_linker
vllm serve "$(python -c 'from plugins.deberta_gliner_linker import get_model_path; print(get_model_path())')" \
  --trust-remote-code --runner pooling
```
