# ColLFM2

Multimodal late-interaction retrieval via LFM2-VL with ColPali-style pooling.

**Model:** [VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1)
**Architecture:** LFM2-VL backbone (Mamba/SSM hybrid), ColPali multi-vector pooling, text + image
**Performance:** vLLM-only (no vanilla baseline available)
**Parity:** cos_sim = 0.9999 (query), 0.9983 (image) vs colpali-engine

## Usage

```python
from vllm import LLM

llm = LLM("VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1", trust_remote_code=True)
outputs = llm.encode(["Describe this document"], pooling_task="token_embed")
```

## Serve

```bash
vllm serve VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1 \
  --trust-remote-code --dtype bfloat16 --port 8200
```

## Verify

```bash
python plugins/collfm2/parity_test.py
```
