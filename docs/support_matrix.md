# Support Matrix

## Core Compatibility

| Component | Supported | Notes |
|---|---|---|
| Python | 3.11, 3.12 | 3.11 is the primary development baseline |
| vLLM | 0.19+ | Native IOProcessor and `extra_kwargs` support; no patching required |
| PyTorch | 2.0+ | Follow vLLM compatibility for pinned versions |
| OS (dev) | Linux, macOS | Production serving requires Linux + NVIDIA GPU |
| GPU backend | CUDA (primary) | macOS/CPU only for development validation |

## Installation

```bash
pip install -e ".[gliner]"    # base deps + GLiNER
pip install vllm               # vLLM >= 0.19 required
```

> No special installation order required. The legacy "install vLLM last" constraint
> from 0.1.x no longer applies.

## Plugin Compatibility

All 12 plugins are tested with `vllm==0.19.0` using the V1 engine with native IOProcessor plugins.

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

## Runtime Monkey-Patches (0.2.0)

Two scoped monkey-patches are applied at model initialization time. Both are idempotent and narrowly targeted.

| Patch | Scope | Affects | Purpose |
|---|---|---|---|
| `GPUModelRunner._preprocess` | `GLiNERLinkerModel.__init__` | linker + rerank plugins | Forwards `attention_mask` from `extra_kwargs` into model forward |
| `Attention.get_kv_cache_spec` | `NemotronColEmbedModel.__init__` | `nemotron_colembed` only | Returns `None` for `ENCODER_ONLY` layers to skip KV cache |

These patches will be removed once vLLM provides native support for forwarding arbitrary `extra_kwargs` keys into model forward and for skipping KV cache on encoder-only attention layers.

## Verify Installation

```bash
python -m vllm_factory.compat.doctor
```

This reports the detected vLLM version, native IO mode, and registered plugins.
