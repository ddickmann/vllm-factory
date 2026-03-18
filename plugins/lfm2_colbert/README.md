# LFM2-ColBERT

Mamba/SSM hybrid backbone with ColBERT multi-vector projection.

**Model:** [LiquidAI/LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M)
**Architecture:** LFM2 (Mamba/SSM + attention hybrid), ColBERT linear projection, L2 normalization
**Performance:** vLLM-only (no vanilla baseline available)
**Parity:** 1.0000 (multi-vector output verified)

## Usage

```python
from vllm import LLM

llm = LLM("LiquidAI/LFM2-ColBERT-350M", trust_remote_code=True)
outputs = llm.encode(["information retrieval with Mamba"], pooling_task="token_embed")
```

## Serve

```bash
vllm serve LiquidAI/LFM2-ColBERT-350M \
  --trust-remote-code --dtype bfloat16 --port 8200
```

## Verify

```bash
python plugins/lfm2_colbert/parity_test.py
```
