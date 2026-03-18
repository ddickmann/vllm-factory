# ModernColBERT

Multi-vector ColBERT retrieval via ModernBERT with custom RoPE kernels.

**Model:** [VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT)
**Architecture:** ModernBERT backbone, ColBERT multi-vector pooling, fused RoPE Triton kernels
**Performance:** Significant speedup vs PyLate — run `benchmark_colbert.py` on your hardware
**Parity:** cos_sim = 0.9998 (query), 0.9999 (doc) vs PyLate

## Usage

```python
from vllm import LLM

llm = LLM("VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT", trust_remote_code=True)
outputs = llm.encode(["What is deep learning?"], pooling_task="token_embed")
# outputs[0].outputs.embedding = list of token-level vectors
```

## Serve

```bash
vllm serve VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT \
  --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --port 8200
```

## Verify

```bash
python plugins/moderncolbert/parity_test.py
```
