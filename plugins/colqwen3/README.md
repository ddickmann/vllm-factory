# ColQwen3

Multimodal late-interaction retrieval via Qwen3-VL with ColPali-style pooling.

**Model:** [VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1)
**Architecture:** Qwen3-VL backbone, ColPali multi-vector pooling, text + image inputs
**Performance:** vLLM-only (no vanilla baseline available)
**Parity:** cos_sim = 0.9997 (query), 0.9900 (image) vs colpali-engine

## Usage

```python
from vllm import LLM

llm = LLM("VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1", trust_remote_code=True)
outputs = llm.encode(["Describe this document"], pooling_task="token_embed")
```

## Serve

```bash
vllm serve VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1 \
  --trust-remote-code --dtype bfloat16 --port 8200
```

## Verify

```bash
python plugins/colqwen3/parity_test.py
```
