# vLLM Factory

**Production inference for encoders, poolers, and structured prediction — as vLLM plugins.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/ddickmann/vllm-factory/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![vLLM 0.15+](https://img.shields.io/badge/vLLM-0.15%2B-green.svg)](https://github.com/vllm-project/vllm)
[![Plugins](https://img.shields.io/badge/Plugins-12_models-purple.svg)](#plugins)
[![Parity](https://img.shields.io/badge/Parity-12%2F12_passing-brightgreen.svg)](#parity)

> **12 encoder plugins · IOProcessor pre/post-processing · continuous batching · zero vLLM forks**

```bash
# Install and serve any model in 3 commands
pip install -e ".[gliner]"
pip install "vllm==0.15.1"          # always install vLLM last

vllm serve VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --io-processor-plugin moderncolbert_io
```

```bash
# Query it
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT",
       "data":{"text":"European Central Bank monetary policy"}}'
```

---

## Why vLLM Factory?

Decoder-based LLM serving is a solved problem. Encoder-based serving is not.

Production traffic is heterogeneous: staggered requests at unpredictable intervals, mixed sequence lengths, variable batch sizes — none of it neatly padded or synchronized. Vanilla PyTorch pipelines (GLiNER, PyLate, SentenceTransformers) process requests sequentially or require manual batching. They block on each `model.forward()`, waste GPU cycles waiting for the next request, and have no scheduler to absorb traffic spikes.

**vLLM Factory bridges that gap.** Every bespoke encoder architecture — ColBERT, GLiNER, entity linking, multimodal retrieval — gets the same production-grade scheduling and memory management as a 70B chat model. No fork. No custom server. Just `vllm serve`.

Each plugin ships an **IOProcessor** that handles all pre- and post-processing inside the vLLM process. Clients send structured JSON (`{"data": {"text": ...}}` or `{"data": {"image": ...}}`), and the IOProcessor converts to model inputs, runs inference, and returns structured results. No client-side tokenization. No manual `extra_kwargs`. Just `POST /pooling`.

| Capability | HF / SentenceTransformers | TEI | **vLLM Factory** |
|---|:---:|:---:|:---:|
| ColBERT multi-vector retrieval | ❌ | ❌ | ✅ |
| GLiNER span-level NER | ❌ | ❌ | ✅ |
| GLiNER2 schema extraction | ❌ | ❌ | ✅ |
| Entity linking + reranking pipeline | ❌ | ❌ | ✅ |
| Multimodal retrieval (ColPali/ColQwen/Nemotron) | ❌ | ❌ | ✅ |
| Continuous batching for encoders | ❌ | ✅ | ✅ |
| CUDA graphs for encoders | ❌ | ✅ | ✅ |
| Built-in pre/post-processing (IOProcessor) | ❌ | ❌ | ✅ |
| Plugin architecture (no fork) | — | — | ✅ |
| End-to-end parity tests | — | — | ✅ |

---

## Installation

```bash
pip install vllm-factory          # from PyPI (Linux, requires CUDA)
```

Or from source for development:

> **Critical: vLLM must be the last package installed.** Other dependencies (especially `gliner`) can pull in `transformers` versions that conflict with vLLM. Installing vLLM last ensures it pins all shared dependencies to compatible versions.

### Standard install

```bash
git clone https://github.com/ddickmann/vllm-factory.git && cd vllm-factory

# Step 1: Install vllm-factory + base dependencies (+ gliner for NER/linking models)
pip install -e ".[gliner]"

# Step 2: Install vLLM — ALWAYS LAST
pip install "vllm==0.15.1"

# Step 3: Apply the pooling patch (one-time, enables extra_kwargs passthrough)
python -m forge.patches.pooling_extra_kwargs
```

### Minimal install (no GLiNER models)

If you only need embedding or ColBERT models (no NER/linking):

```bash
pip install -e .
pip install "vllm==0.15.1"
python -m forge.patches.pooling_extra_kwargs
```

### Docker

```dockerfile
FROM vllm/vllm-openai:v0.15.1

COPY . /app/vllm-factory
WORKDIR /app/vllm-factory

# Install deps first, vLLM is already in base image (last)
RUN pip install -e ".[gliner]"
RUN python -m forge.patches.pooling_extra_kwargs

CMD ["vllm", "serve", "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT", \
     "--runner", "pooling", "--trust-remote-code", "--dtype", "bfloat16", \
     "--io-processor-plugin", "moderncolbert_io"]
```

### Verify installation

```bash
make test-serve P=embeddinggemma   # Fastest model — starts server, runs test, reports pass/fail
```

---

## Serving — all 12 models

Every plugin is served with `vllm serve` + `--io-processor-plugin`. The IOProcessor handles all tokenization, formatting, and output decoding server-side. Clients send simple JSON.

### Embedding

**EmbeddingGemma** — dense CLS embeddings (300M)

```bash
vllm serve unsloth/embeddinggemma-300m \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching \
  --io-processor-plugin embeddinggemma_io
```

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth/embeddinggemma-300m",
       "data":{"text":"What is the knapsack problem?"}}'
```

### Late Interaction / Retrieval

**ModernColBERT** — multi-vector ColBERT (ModernBERT backbone)

```bash
vllm serve VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin moderncolbert_io
```

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT",
       "data":{"text":"European Central Bank monetary policy"}}'
```

**LFM2-ColBERT** — Mamba/SSM hybrid ColBERT (350M)

```bash
vllm serve LiquidAI/LFM2-ColBERT-350M \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin lfm2_colbert_io
```

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"LiquidAI/LFM2-ColBERT-350M",
       "data":{"text":"Mamba state-space model architecture"}}'
```

### Multimodal Retrieval (text + vision)

**ColQwen3** — Qwen3-VL + ColPali (1.7B)

```bash
vllm serve VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1 \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --max-model-len 8192 --limit-mm-per-prompt '{"image": 1}' \
  --io-processor-plugin colqwen3_io
```

```bash
# Text query
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1",
       "data":{"text":"What does the revenue chart show?", "is_query": true}}'

# Image document
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1",
       "data":{"image":"https://example.com/document.png", "is_query": false}}'
```

**ColLFM2** — LFM2-VL + ColPali (450M, multimodal)

```bash
vllm serve VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1 \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin collfm2_io
```

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1",
       "data":{"text":"Summarize the table contents"}}'
```

**Nemotron-ColEmbed** — bidirectional Qwen3-VL (4B, multimodal)

```bash
vllm serve nvidia/nemotron-colembed-vl-4b-v2 \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin nemotron_colembed_io
```

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/nemotron-colembed-vl-4b-v2",
       "data":{"text":"Neural network optimization techniques", "is_query": true}}'
```

### Named Entity Recognition (GLiNER)

GLiNER models use custom model directories prepared by `forge/model_prep.py`. The IOProcessor handles all NER preprocessing (tokenization, span generation) and postprocessing (entity decoding) server-side.

> Requires `pip install -e ".[gliner]"` at install time.

**mmbert_gliner** — ModernBERT + GLiNER span head

```bash
# Prepare model (one-time)
vllm-factory-prep --model VAGOsolutions/SauerkrautLM-GLiNER --output /tmp/sauerkraut-gliner-vllm

# Serve
vllm serve /tmp/sauerkraut-gliner-vllm \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin mmbert_gliner_io
```

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"/tmp/sauerkraut-gliner-vllm",
       "data":{
         "text":"Apple Inc. announced a partnership with OpenAI. Tim Cook presented at WWDC 2024.",
         "labels":["company","person","event"],
         "threshold":0.3
       }}'
```

Returns: `{"data": [{"text": "Apple Inc.", "label": "company", "score": 0.95}, ...]}`

**mt5_gliner** — mT5 encoder + multilingual GLiNER

```bash
vllm-factory-prep --model knowledgator/gliner-x-large --output /tmp/gliner-x-large-vllm

vllm serve /tmp/gliner-x-large-vllm \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin mt5_gliner_io
```

**deberta_gliner** — DeBERTa v2 + GLiNER span head

```bash
vllm-factory-prep --model urchade/gliner_small-v2.1 --output /tmp/gliner-pii-vllm

vllm serve /tmp/gliner-pii-vllm \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin deberta_gliner_io
```

**deberta_gliner2** — DeBERTa v3 + GLiNER2 schema extraction

```bash
vllm-factory-prep --model fastino/gliner2-large-v1 --output /tmp/gliner2-vllm

vllm serve /tmp/gliner2-vllm \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin deberta_gliner2_io
```

### Entity Linking & Reranking

**deberta_gliner_linker** — dual DeBERTa + LSTM + scorer (L3)

```bash
vllm serve plugins/deberta_gliner_linker/_model_cache \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin deberta_gliner_linker_io
```

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model":"plugins/deberta_gliner_linker/_model_cache",
       "data":{
         "text":"Tesla announced record earnings in Austin.",
         "labels":["company","location"],
         "threshold":0.3,
         "candidate_labels":["Tesla Inc.","Austin, TX","TSLA"]
       }}'
```

**modernbert_gliner_rerank** — ModernBERT + projection + LSTM + scorer (L4)

```bash
vllm serve plugins/modernbert_gliner_rerank/_model_cache \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --io-processor-plugin modernbert_gliner_rerank_io
```

---

## Plugins

### Embedding

| Plugin | Architecture | Checkpoint | Params |
|---|---|---|---|
| `embeddinggemma` | Gemma + CLS projection | [`unsloth/embeddinggemma-300m`](https://huggingface.co/unsloth/embeddinggemma-300m) | 300M |

### Late Interaction / Retrieval

| Plugin | Architecture | Checkpoint | Params |
|---|---|---|---|
| `moderncolbert` | ModernBERT + ColBERT | [`VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT`](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT) | 149M |
| `lfm2_colbert` | LFM2 (Mamba/SSM) + ColBERT | [`LiquidAI/LFM2-ColBERT-350M`](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M) | 350M |
| `colqwen3` | Qwen3-VL + ColPali (vision) | [`VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1`](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1) | 1.7B |
| `collfm2` | LFM2-VL + ColPali (vision) | [`VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1`](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1) | 450M |
| `nemotron_colembed` | Qwen3-VL bidirectional + ColBERT | [`nvidia/nemotron-colembed-vl-4b-v2`](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) | 4B |

### Named Entity Recognition (GLiNER)

| Plugin | Architecture | Checkpoint | Params |
|---|---|---|---|
| `mmbert_gliner` | ModernBERT + GLiNER span head | [`VAGOsolutions/SauerkrautLM-GLiNER`](https://huggingface.co/VAGOsolutions/SauerkrautLM-GLiNER) | 150M |
| `deberta_gliner` | DeBERTa v2 + GLiNER span head | [`urchade/gliner_small-v2.1`](https://huggingface.co/urchade/gliner_small-v2.1) | 166M |
| `mt5_gliner` | mT5 encoder + multilingual GLiNER | [`knowledgator/gliner-x-large`](https://huggingface.co/knowledgator/gliner-x-large) | 800M |
| `deberta_gliner2` | DeBERTa v3 + GLiNER2 schema extraction | [`fastino/gliner2-large-v1`](https://huggingface.co/fastino/gliner2-large-v1) | 304M |

### Entity Linking & Reranking

| Plugin | Architecture | Checkpoint | Params |
|---|---|---|---|
| `deberta_gliner_linker` | Dual DeBERTa + LSTM + scorer | [`knowledgator/gliner-linker-large-v1.0`](https://huggingface.co/knowledgator/gliner-linker-large-v1.0) | 304M |
| `modernbert_gliner_rerank` | ModernBERT + projection + LSTM | [`knowledgator/gliner-linker-rerank-v1.0`](https://huggingface.co/knowledgator/gliner-linker-rerank-v1.0) | 68M |

---

## Parity — all 12 plugins validated

Every plugin passes end-to-end parity testing: `vllm serve` → HTTP request → compare against reference implementation. No smoke tests — real model inference, real outputs.

**NER models** are validated by comparing actual entity text and labels (not counts). The gating metric is **recall** — every reference entity must be found by vLLM. vLLM finding extra entities is acceptable. Entity confidence scores are compared informally (score deltas reported but not gating, since dtype rounding produces small drift).

**Embedding/ColBERT models** are validated by element-wise cosine similarity of the full output vector against reference tensors from the vanilla library.

All models run in **bfloat16**.

| Plugin | Reference | Metric | Score |
|---|---|---|---|
| `embeddinggemma` | HF SentenceTransformer | cosine sim | **1.0000** |
| `mmbert_gliner` | GLiNER library | recall (entity text+label) | **1.000** |
| `deberta_gliner` | GLiNER library | recall (entity text+label) | **1.000** |
| `deberta_gliner2` | GLiNER2 library | recall (entity text+label) | **1.000** |
| `mt5_gliner` | GLiNER library | recall (entity text+label) | **1.000** |
| `deberta_gliner_linker` | Knowledgator GLinker | recall + link match | **1.000** |
| `modernbert_gliner_rerank` | Knowledgator GLinker | recall (entity text+label) | **1.000** |
| `moderncolbert` | PyLate | cosine sim | **0.970** |
| `lfm2_colbert` | HF transformers | cosine sim | **1.000** |
| `collfm2` | sauerkrautlm-colpali | cosine sim | **0.9996** |
| `colqwen3` | sauerkrautlm-colpali | cosine sim | **0.9966** |
| `nemotron_colembed` | HF transformers | cosine sim | **0.9997** |

```bash
python scripts/serve_parity_test.py                # all 12 plugins
python scripts/serve_parity_test.py --plugin colqwen3  # single plugin
```

---

## How it works

### IOProcessor architecture

Each plugin registers an **IOProcessor** — a vLLM-native plugin that runs pre/post-processing inside the serving process. No client-side tokenization needed.

```
POST /pooling {"data": {"text": "..."}}
    │
    ▼
┌─────────────────────────────────────────────────┐
│  IOProcessor.parse_request()  → typed input      │
│  IOProcessor.pre_process()    → tokenized prompt  │
│  engine.encode()              → model forward     │
│  IOProcessor.post_process()   → structured output │
│  IOProcessor.output_to_response() → JSON response │
└─────────────────────────────────────────────────┘
    │
    ▼
{"data": [{"text": "Apple Inc.", "label": "company", "score": 0.95}, ...]}
```

### Custom Triton Kernels

| Kernel | What it optimizes |
|---|---|
| `flash_deberta_attention` | Fused c2p + p2c disentangled relative position bias for DeBERTa |
| `fused_glu_mlp` | Fused GeGLU chunk + GELU + mul + dropout |
| `fused_rope_global` | RoPE for ModernBERT global attention layers |
| `fused_rope_local` | RoPE for ModernBERT sliding-window local attention |
| `fused_layernorm` | Single-pass mean/var/normalize + affine |
| `fused_dropout_residual` | In-place dropout + residual add |

### Repository structure

```
vllm-factory/
├── plugins/              # 12 model plugins (each with io_processor.py + parity_test.py)
├── models/               # Encoder backbones (DeBERTa, ModernBERT, mT5, ...)
├── kernels/              # Custom Triton kernels
├── poolers/              # Shared pooler heads (ColBERT, GLiNER, ColPali, linker)
├── forge/                # Shared infrastructure (model_prep, patches, server utilities)
├── examples/             # Ready-to-run example scripts
├── scripts/              # Parity test orchestrator, reference generators
├── notebooks/            # Jupyter notebooks for each model family
├── Makefile              # install · serve · test · bench · lint
└── pyproject.toml        # All 12 plugins registered as vLLM entry points
```

---

## Building custom plugins

See [`docs/PLUGIN_GUIDE.md`](docs/PLUGIN_GUIDE.md) for the step-by-step walkthrough.

A new plugin needs:

| File | Purpose |
|---|---|
| `config.py` | HuggingFace-compatible config (dimensions, layers) |
| `model.py` | Encoder forward path + `self.pooler` wiring |
| `io_processor.py` | IOProcessor — parse, pre-process, post-process, response |
| `parity_test.py` | Validation against reference implementation |

---

## Why it's fast

**Vanilla PyTorch blocks.** One `model.forward()` at a time. If request B arrives while request A is mid-inference, B waits. Under staggered, heterogeneous load — which is what production actually looks like — GPU utilization craters.

**vLLM schedules.** Incoming requests are continuously batched by the async scheduler. Variable-length sequences are packed efficiently via PagedAttention. CUDA graphs eliminate kernel launch overhead. The GPU stays saturated regardless of arrival pattern.

vLLM Factory brings this to every encoder architecture with zero custom serving code.

#### Measured speedups (RTX 4090, 124 requests, 512 tokens)

| Model | Vanilla | vLLM Factory | Speedup |
|---|---|---|---|
| **LFM2-ColBERT** (350M, Mamba/SSM) | HF AutoModel | `vllm serve` | **6.7×** |
| **MT5 GLiNER** (800M, NER) | GLiNER lib | `vllm serve` | **2.7×** |
| **EmbeddingGemma** (300M, dense) | SentenceTransformers | `vllm serve` | **1.8×** |

---

## Design principles

- **No vLLM forks** — plugins, not patches
- **Parity before performance** — every optimization validated against reference
- **IOProcessor-first** — all pre/post-processing runs server-side
- **vLLM must install last** — dependency order is enforced to avoid version conflicts
- **Task-aware architecture** — backbone + pooler + IOProcessor = single deployment contract

---

## Requirements

- Python 3.11+
- PyTorch 2.0+
- vLLM 0.15+ (installed last)
- NVIDIA GPU with CUDA support (production)
- Triton 2.0+ (for custom kernels, optional)
- macOS users: see [`docs/macos_vllm.md`](docs/macos_vllm.md) for local dev setup (CPU only, no production serving)

## Enterprise support

Running vLLM Factory in production? [Latence AI](https://latence.ai) provides custom plugin development, performance optimization, and deployment review.

**→ [hello@latence.ai](mailto:hello@latence.ai) · [GitHub Issues](https://github.com/ddickmann/vllm-factory/issues)**

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

```bash
make install       # install everything (correct dep order)
make serve P=name  # serve a plugin
make test P=name   # run parity test
make lint          # ruff check
```

## Acknowledgements

| Project | Authors | Contribution |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | vLLM Team | High-throughput serving engine |
| [GLiNER](https://github.com/urchade/GLiNER) | Urchade Zaratiana et al. | Generalist NER architecture |
| [FlashDeBERTa](https://github.com/Knowledgator/FlashDeBERTa) | Knowledgator | Triton kernel for DeBERTa attention |
| [GLinker](https://github.com/Knowledgator/GLinker) | Knowledgator | Entity linking architecture |
| [PyLate](https://github.com/lightonai/pylate) | LightOn AI | ColBERT training/inference reference |
| [sauerkrautlm-colpali](https://github.com/VAGOsolutions/sauerkrautlm-colpali) | VAGO Solutions | ColQwen/ColPali models |
| [NV-Retriever](https://arxiv.org/abs/2602.03992) | NVIDIA | Nemotron-ColEmbed architecture |
| [LFM2](https://www.liquid.ai/) | Liquid AI | LFM2 Mamba/SSM hybrid models |
| [ColBERT](https://github.com/stanford-futuredata/ColBERT) | Omar Khattab (Stanford) | Late-interaction retrieval paradigm |
| [ColPali](https://arxiv.org/abs/2407.01449) | Illuin Technology | Vision-language retrieval |
| [ModernBERT](https://arxiv.org/abs/2412.13663) | Answer.AI & LightOn | Modern BERT architecture |

## License

Apache 2.0
