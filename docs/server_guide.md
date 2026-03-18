# Server Deployment Guide

> **vLLM Factory is designed for `vllm serve`.** All 12 plugins are tested and documented for server-mode deployment with IOProcessor plugins.

## Architecture

Each plugin registers two entry points:
1. **`vllm.general_plugins`** — registers the model architecture (backbone + pooler)
2. **`vllm.io_processor_plugins`** — registers the IOProcessor (pre/post-processing)

When you run `vllm serve --io-processor-plugin <name>`, the IOProcessor handles tokenization, prompt formatting, and output decoding inside the server process. Clients send simple JSON, not raw token IDs.

## Quick Server Setup

### CLI (Simplest)

```bash
vllm serve VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT \
  --runner pooling \
  --trust-remote-code \
  --dtype bfloat16 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --io-processor-plugin moderncolbert_io \
  --port 8000
```

### Using Makefile

```bash
make serve P=moderncolbert PORT=8000
```

### Docker (Production)

```dockerfile
FROM vllm/vllm-openai:v0.15.1

COPY . /app/vllm-factory
WORKDIR /app/vllm-factory

# Install deps first — vLLM is already in the base image (installed last)
RUN pip install -e ".[gliner]"
RUN python -m forge.patches.pooling_extra_kwargs

EXPOSE 8000
CMD ["vllm", "serve", "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT", \
     "--runner", "pooling", "--trust-remote-code", "--dtype", "bfloat16", \
     "--no-enable-prefix-caching", "--no-enable-chunked-prefill", \
     "--io-processor-plugin", "moderncolbert_io", "--port", "8000"]
```

## Request Format

All IOProcessor plugins accept the same request structure via `POST /pooling`:

```json
{
  "model": "<model-name-or-path>",
  "data": {
    "text": "Your input text",
    ...additional fields per plugin...
  }
}
```

### Embedding models

```json
{"model": "unsloth/embeddinggemma-300m", "data": {"text": "Hello world"}}
```

### ColBERT / multi-vector models

```json
{"model": "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT", "data": {"text": "query text"}}
```

### Multimodal models (text or image)

```json
{"model": "...", "data": {"text": "What does this show?", "is_query": true}}
{"model": "...", "data": {"image": "https://example.com/doc.png", "is_query": false}}
```

Image inputs accept: URLs (`https://...`), base64 data URIs (`data:image/png;base64,...`), or local file paths.

### GLiNER NER models

```json
{
  "model": "/tmp/sauerkraut-gliner-vllm",
  "data": {
    "text": "Apple announced a partnership with OpenAI.",
    "labels": ["company", "product"],
    "threshold": 0.3,
    "flat_ner": true
  }
}
```

### Entity linking models

```json
{
  "model": "plugins/deberta_gliner_linker/_model_cache",
  "data": {
    "text": "Tesla announced record earnings.",
    "labels": ["company", "location"],
    "threshold": 0.3,
    "candidate_labels": ["Tesla Inc.", "Austin, TX"]
  }
}
```

## Key Server Arguments

| Argument | Description |
|----------|-------------|
| `--runner pooling` | Required for all encoder/embedding models |
| `--trust-remote-code` | Required for custom configs |
| `--io-processor-plugin <name>` | Activates the IOProcessor for pre/post-processing |
| `--dtype bfloat16` | Recommended for all models |
| `--no-enable-prefix-caching` | Required for encoder models |
| `--no-enable-chunked-prefill` | Recommended for encoder models |
| `--gpu-memory-utilization` | GPU memory fraction (default 0.9) |
| `--max-model-len` | Override max sequence length (needed for ColQwen3: 8192) |
| `--enforce-eager` | Disable CUDA graphs (for debugging) |
| `--port` / `--uds` | TCP port or Unix socket path |

## Multi-Model on One GPU

Run multiple models on different ports:

```bash
# Terminal 1: ColBERT
vllm serve VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin moderncolbert_io \
  --gpu-memory-utilization 0.3 --port 8001

# Terminal 2: GLiNER
vllm serve /tmp/sauerkraut-gliner-vllm \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin mmbert_gliner_io \
  --gpu-memory-utilization 0.3 --port 8002
```

## Health Checks

```bash
curl http://localhost:8000/health
```

Returns `200 OK` when the model is loaded and ready.

## IOProcessor Plugin Names

| Plugin | IOProcessor Entry Point |
|---|---|
| `embeddinggemma` | `embeddinggemma_io` |
| `moderncolbert` | `moderncolbert_io` |
| `lfm2_colbert` | `lfm2_colbert_io` |
| `colqwen3` | `colqwen3_io` |
| `collfm2` | `collfm2_io` |
| `nemotron_colembed` | `nemotron_colembed_io` |
| `mmbert_gliner` | `mmbert_gliner_io` |
| `deberta_gliner` | `deberta_gliner_io` |
| `mt5_gliner` | `mt5_gliner_io` |
| `deberta_gliner2` | `deberta_gliner2_io` |
| `deberta_gliner_linker` | `deberta_gliner_linker_io` |
| `modernbert_gliner_rerank` | `modernbert_gliner_rerank_io` |
