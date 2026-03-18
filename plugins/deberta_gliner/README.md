# DeBERTa-GLiNER

Named entity recognition via DeBERTa v2 + GLiNER span scoring head.

**Model:** [nvidia/gliner-PII](https://huggingface.co/nvidia/gliner-PII)
**Architecture:** DeBERTa v2 backbone with flash DeBERTa Triton kernel, GLiNER span-logit pooler
**Performance:** Competitive with vanilla GLiNER; benefits grow at higher concurrency — run `benchmark_gliner.py` on your hardware
**Parity:** Entity F1 = 1.0000 (7/7 entities)

> DeBERTa v2's disentangled attention adds overhead in the vLLM pipeline that exceeds the batching benefit at low concurrency. The vanilla GLiNER library uses Flash Attention natively. At higher concurrency, vLLM's scheduling advantage grows.

## Usage

```python
from vllm import LLM

llm = LLM("nvidia/gliner-PII", trust_remote_code=True)
# Use with GLiNER processor for entity extraction — see notebooks/02_gliner_ner.ipynb
```

## Serve

Requires a prepared model directory (see `forge/model_prep.py`).

```bash
vllm serve /tmp/gliner-pii-vllm \
  --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --port 8200
```

## Verify

```bash
python plugins/deberta_gliner/parity_test.py
```
