# ModernBERT-GLiNER

Named entity recognition via ModernBERT + GLiNER span scoring head.

**Model:** [VAGOsolutions/SauerkrautLM-GLiNER](https://huggingface.co/VAGOsolutions/SauerkrautLM-GLiNER)
**Architecture:** ModernBERT backbone with fused RoPE kernels, GLiNER span-logit pooler
**Performance:** Significant speedup vs vanilla GLiNER — run `benchmark_gliner.py` on your hardware
**Parity:** Entity F1 = 1.0000 (11/11 entities)

## Usage

```python
from vllm import LLM

llm = LLM("VAGOsolutions/SauerkrautLM-GLiNER", trust_remote_code=True)
# Use with GLiNER processor for entity extraction — see notebooks/02_gliner_ner.ipynb
```

## Serve

Requires a prepared model directory (see `forge/model_prep.py`).

```bash
vllm serve /tmp/sauerkraut-gliner-vllm \
  --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --port 8200
```

## Verify

```bash
python plugins/mmbert_gliner/parity_test.py
```
