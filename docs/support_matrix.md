# Support Matrix

## Core Compatibility

| Component | Supported | Notes |
|---|---|---|
| Python | 3.11, 3.12 | 3.11 is the primary development baseline |
| vLLM | 0.15.x | IOProcessor plugins require 0.15.0+; `forge/patches/pooling_extra_kwargs.py` targets 0.15.x |
| PyTorch | 2.0+ | Follow vLLM compatibility for pinned versions |
| OS (dev) | Linux, macOS | Production serving requires Linux + NVIDIA GPU |
| GPU backend | CUDA (primary) | macOS/CPU only for development validation |

## Installation Order

> **vLLM must be the last package installed.** Other dependencies (especially `gliner`) can pull in `transformers` versions that conflict with vLLM.

```bash
pip install -e ".[gliner]"    # base deps + GLiNER
pip install "vllm==0.15.1"    # vLLM — always last
```

## Plugin Compatibility

All 12 plugins are tested with `vllm==0.15.1` using IOProcessor plugins.

| Plugin | IOProcessor | dtype | Notes |
|---|---|---|---|
| `embeddinggemma` | `embeddinggemma_io` | bfloat16 | — |
| `moderncolbert` | `moderncolbert_io` | bfloat16 | Requires `--no-enable-prefix-caching --no-enable-chunked-prefill` |
| `lfm2_colbert` | `lfm2_colbert_io` | bfloat16 | Requires `--no-enable-prefix-caching --no-enable-chunked-prefill` |
| `colqwen3` | `colqwen3_io` | bfloat16 | Requires `--max-model-len 8192 --limit-mm-per-prompt '{"image": 1}'` |
| `collfm2` | `collfm2_io` | bfloat16 | Requires `--no-enable-prefix-caching --no-enable-chunked-prefill` |
| `nemotron_colembed` | `nemotron_colembed_io` | bfloat16 | 4B model, requires ~16GB VRAM |
| `mmbert_gliner` | `mmbert_gliner_io` | bfloat16 | Requires `pip install -e ".[gliner]"` |
| `deberta_gliner` | `deberta_gliner_io` | bfloat16 | Requires `pip install -e ".[gliner]"` |
| `mt5_gliner` | `mt5_gliner_io` | bfloat16 | Requires `pip install -e ".[gliner]"` |
| `deberta_gliner2` | `deberta_gliner2_io` | bfloat16 | — |
| `deberta_gliner_linker` | `deberta_gliner_linker_io` | bfloat16 | Requires `pip install -e ".[gliner]"` |
| `modernbert_gliner_rerank` | `modernbert_gliner_rerank_io` | bfloat16 | Requires `pip install -e ".[gliner]"`, uses custom ModernBERT encoder |

## Patch Compatibility

The pooling patch (`forge/patches/pooling_extra_kwargs.py`) targets vLLM `>=0.15.0` and `<0.16.0`. It enables `extra_kwargs` passthrough and nested tensor responses for custom poolers.

To force on unsupported versions (for experiments only):

```bash
VLLM_FACTORY_ALLOW_UNSUPPORTED_VLLM=1 python -m forge.patches.pooling_extra_kwargs
```
