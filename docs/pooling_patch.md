# Pooling Patch — Removed in 0.2.0

> **This patch was removed in vllm-factory 0.2.0.**
>
> vLLM >= 0.19 supports `extra_kwargs` passthrough and custom IOProcessors
> natively. No disk-patching of vLLM source files is required.

## What Changed

The legacy patch (`forge/patches/pooling_extra_kwargs.py`) modified vLLM's
`pooling/protocol.py` at install time to:

1. Add `extra_kwargs` field to `PoolingCompletionRequest`
2. Pass `extra_kwargs` through to `PoolingParams`
3. Accept nested (3D+) tensor responses in `PoolingResponseData`

All three capabilities are now provided by vLLM >= 0.19 out of the box.

## Migration

If upgrading from vllm-factory 0.1.x:

```bash
pip install vllm>=0.19        # native support, no patching
pip install -e ".[gliner]"    # reinstall vllm-factory
python -m vllm_factory.compat.doctor   # verify: should show "NATIVE IO PROCESSOR"
```

No other changes are needed. All 12 plugins use the native `IOProcessor` path automatically.

## See Also

- [CHANGELOG.md](../CHANGELOG.md) — 0.2.0 release notes
- [docs/support_matrix.md](support_matrix.md) — current compatibility matrix
- [docs/architecture.md](architecture.md) — updated architecture overview
