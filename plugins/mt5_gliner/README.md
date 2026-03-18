# mT5-GLiNER

Multilingual named entity recognition via mT5 encoder + GLiNER span scoring.

**Model:** [knowledgator/gliner-x-large](https://huggingface.co/knowledgator/gliner-x-large)
**Architecture:** mT5 encoder backbone, GLiNER span-logit pooler, multilingual support
**Performance:** Speedup vs vanilla GLiNER, especially at higher batch sizes — run `benchmark_gliner.py` on your hardware
**Parity:** Entity F1 = 1.0000 (11/11 entities)

## Usage

```python
from vllm import LLM

llm = LLM("knowledgator/gliner-x-large", trust_remote_code=True)
# Use with GLiNER processor for multilingual entity extraction
```

## Serve

Requires a prepared model directory (see `forge/model_prep.py`).

```bash
vllm serve /tmp/gliner-x-large-vllm \
  --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --port 8200
```

## Verify

```bash
python plugins/mt5_gliner/parity_test.py
```
