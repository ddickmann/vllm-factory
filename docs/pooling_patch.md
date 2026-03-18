# Pooling Patch for vLLM 0.15.x

> **Note:** With IOProcessor plugins, most of the pre/post-processing happens inside the IOProcessor. This patch is still needed for `extra_kwargs` passthrough to custom poolers and for 3D+ tensor response support.

## Why Is This Needed?

vLLM's built-in `/pooling` HTTP endpoint has two gaps that affect custom pooling models:

### 1. `extra_kwargs` Are Silently Dropped

`PoolingParams` supports `extra_kwargs` — a dict of structured metadata that custom poolers use:
- **GLiNER**: `entity_spans`, `words_mask`, `entities`
- **Span Predictor**: `span_tokens`, `candidate_entities`
- **ColBERT**: `query_length`, `is_query`

But the HTTP request/response protocol classes don't expose this field. Metadata sent via HTTP is silently dropped, so your custom pooler receives `None` instead of the structured data it needs.

### 2. 3D+ Tensor Responses Fail Validation

Custom models return multi-dimensional tensors:
- **GLiNER**: `(L, max_width, num_classes)` — 3D logits
- **Span Predictor**: `(L, max_width)` — 2D logits

vLLM's `PoolingResponseData.data` validates as `list[list[float]] | list[float] | str`, rejecting anything deeper than 2D with a Pydantic validation error.

## How the Patch Works

The patch modifies `vllm/entrypoints/pooling/pooling/protocol.py` in place:

1. Adds `extra_kwargs: dict[str, Any] | None` field to both `PoolingCompletionRequest` and `PoolingChatRequest`
2. Passes `extra_kwargs` through to `PoolingParams` in `to_pooling_params()`
3. Changes `PoolingResponseData.data` type from `list[list[float]] | list[float] | str` to `Any`

The patch is **idempotent** — safe to run multiple times.

## Reliability Hardening

- The patcher uses class-aware matching (not one exact multiline snippet), so it tolerates reasonable formatting/layout differences in vLLM 0.15.x protocol files.
- It validates the installed vLLM version range before mutation.
- It self-verifies both behaviors after patching:
  - request `extra_kwargs` passthrough
  - nested (3D) pooling response data acceptance

## Usage

### In Dockerfile (Recommended)

```dockerfile
# After installing vLLM:
RUN python -m forge.patches.pooling_extra_kwargs
```

### From Shell

```bash
python -m forge.patches.pooling_extra_kwargs
```

### From Python

```python
from forge.patches.pooling_extra_kwargs import apply_patch
apply_patch()
```

## When to Re-Apply

Re-apply the patch after:
- Installing or upgrading vLLM
- Reinstalling vLLM from source

The patch self-verifies — if it prints `[PATCH] Verify: ... OK` for both checks, you're good.

## Version-Specific

This patch targets **vLLM 0.15.x**. The protocol file location and structure may change in future versions. If the patch reports "pattern not found", the vLLM version may have changed the protocol structure — check the [changelog](https://docs.vllm.ai/en/latest/) and file an issue.

The patch script validates installed vLLM version before applying changes.
To bypass this safety check for experiments only:

```bash
VLLM_FACTORY_ALLOW_UNSUPPORTED_VLLM=1 python -m forge.patches.pooling_extra_kwargs
```

Use the override only for temporary local experiments.

Upstream tracking issue: [vllm-project/vllm#37344](https://github.com/vllm-project/vllm/issues/37344)
