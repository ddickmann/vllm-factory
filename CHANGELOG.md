# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

## [0.2.0] - 2026-04-05

### Changed
- **vLLM 0.19 native support** — all 12 plugins now work with vLLM >= 0.19 out of the box via the V1 engine
- Introduced `FactoryPooler` protocol (`vllm_factory/pooling/protocol.py`) — pooler business logic is fully decoupled from vLLM internals (zero vLLM imports)
- Introduced `VllmPoolerAdapter` (`vllm_factory/pooling/vllm_adapter.py`) — single file bridging `FactoryPooler` to vLLM's Pooler ABC
- Introduced `FactoryIOProcessor` base class (`vllm_factory/io/base.py`) — single adapter between vLLM's `IOProcessor` ABC and plugin I/O logic; all 12 plugins inherit from it
- All `print()` calls in plugin model files replaced with `logging.getLogger(__name__)`
- Benchmark methodology: vanilla baselines now use `batch_size=1` for fair single-request comparison
- Minimum vLLM version raised from 0.15.x to 0.19+

### Removed
- Legacy vLLM 0.15.x disk-patching mechanism (`forge/patches/pooling_extra_kwargs.py`)
- `Legacy015PatchBridge` and `needs_legacy_patch` capability flag
- `vllm_factory/compat/legacy_patch.py` module
- `tests/test_pooling_patch.py`
- "Install vLLM last" requirement — no longer necessary with native IOProcessor support

### Fixed
- GLiNER Linker entity score parity — integrated `span_rep_layer` for correct `span_logits` computation
- GLiNER Linker concurrency crash (`AssertionError: assert req_state.detokenizer is not None`)
- Race condition in `_pending_extra` request metadata storage (now dict-keyed by `request_id`)
- ModernColBERT `max_position_embeddings` handling for query vs document paths

### Known Limitations
- 2 scoped monkey-patches remain (both idempotent, applied in model `__init__`, documented in-code):
  1. `GPUModelRunner._preprocess` — forwards `attention_mask` from `extra_kwargs` into model forward (required by GLiNER linker/rerank plugins)
  2. `Attention.get_kv_cache_spec` — returns `None` for `ENCODER_ONLY` attention layers to skip unnecessary KV cache allocation (required by `nemotron_colembed`)
- GLiNER models show 10–30% throughput reduction at high concurrency (c≥32) compared to 0.1.x on vLLM 0.15.1, attributed to vLLM 0.19 V1 engine IPC overhead for `extra_kwargs` payloads
- `colqwen3` and `nemotron_colembed` were verified for model loading and parity but not full-benchmark throughput tested in this release cycle

## [0.1.0] - 2026-03-31

### Added
- 12 production-ready vLLM plugin entry points for encoder/pooler workloads.
- IOProcessor integration for all plugins (server-side pre/post-processing via `/pooling` endpoint).
- Custom ModernBERT encoder with dual RoPE, SDPA attention, and block-diagonal masking.
- Custom Triton kernels for flash attention with relative position bias.
- Shared GLiNER preprocessor/postprocessor with batched tokenization support.
- Label embedding caching in linker IOProcessor for reduced per-request overhead.
- Configurable `max_num_batched_tokens` in `forge/processor_base.py`.
- End-to-end parity testing via `vllm serve` + HTTP requests for all 12 plugins.
- Recall-gated NER validation with informational score parity reporting.
- All models validated in bfloat16 with recall=1.0 for NER and cosine >= 0.95 for embeddings.
- CI workflow with lint, import smoke checks, and CPU-safe tests.
- Release workflow with PyPI trusted publisher (OIDC) auto-publish on tag push.
- Issue templates (bug report, feature request, roadmap item).
- Developer workflow documentation with release checklist.
- Support matrix, quickstart guide, server guide, and plugin development guide.
