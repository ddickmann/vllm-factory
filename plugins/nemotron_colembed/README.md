# Nemotron-ColEmbed

Bidirectional Qwen3-VL (4B) for multi-vector ColBERT + vision embeddings.

**Model:** [nvidia/nemotron-colembed-vl-4b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2)
**Architecture:** Qwen3-VL with bidirectional attention, L2-normalized token-level output, no final RMSNorm
**Performance:** vLLM-only (no vanilla baseline available)
**Parity:** cos_sim = 0.9999 (query), 0.9996 (image)

## Usage

```python
from vllm import LLM

llm = LLM("nvidia/nemotron-colembed-vl-4b-v2", trust_remote_code=True)
outputs = llm.encode(["deep learning fundamentals"], pooling_task="token_embed")
```

## Serve

```bash
vllm serve nvidia/nemotron-colembed-vl-4b-v2 \
  --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --port 8200
```

## Verify

```bash
python plugins/nemotron_colembed/parity_test.py
```
