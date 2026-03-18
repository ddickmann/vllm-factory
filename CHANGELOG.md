# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

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
