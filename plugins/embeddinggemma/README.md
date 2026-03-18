# EmbeddingGemma

Dense text embeddings via Gemma with MEAN pooling + learned projection.

**Model:** [unsloth/embeddinggemma-300m](https://huggingface.co/unsloth/embeddinggemma-300m)
**Architecture:** Gemma backbone, MEAN pooling, learned linear projection
**Performance:** Competitive with SentenceTransformers, continuous batching enabled — run `benchmark_all.py` on your hardware
**Parity:** cos_sim = 0.9999 vs HF SentenceTransformer

## Usage

```python
from vllm import LLM

llm = LLM("unsloth/embeddinggemma-300m", trust_remote_code=True)
outputs = llm.embed(["What is the knapsack problem?"])
embedding = outputs[0].outputs.embedding  # dense vector
```

## Serve

```bash
vllm serve unsloth/embeddinggemma-300m \
  --trust-remote-code --dtype bfloat16 --port 8200
```

## Verify

```bash
python plugins/embeddinggemma/parity_test.py
```
