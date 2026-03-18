# IO Processor Pilot — Validation Results

Pilot plugin: `mmbert_gliner` (ModernBERT + GLiNER span extraction)
vLLM version: `0.15.1`
Date: 2026-03-30

## 1. Parity Quality

### Test Results (all PASS)

| Test | Result | Details |
|------|--------|---------|
| Entry-point discovery | PASS | `mmbert_gliner_io` found in `vllm.io_processor_plugins` group |
| Class resolution | PASS | `resolve_obj_by_qualname` resolves `MMBertGLiNERIOProcessor` correctly |
| ABC contract | PASS | All 4 abstract methods implemented: `parse_request`, `pre_process`, `post_process`, `output_to_response` |
| parse_request validation | PASS | Handles dict input, object input, and rejects empty labels |
| validate_or_generate_params | PASS | Stash/pop pattern correctly transfers extra_kwargs between pre_process and params |
| Response contract | PASS | `output_to_response` returns valid `IOProcessorResponse` with preserved data |
| Pre-processing parity | PASS | Identical token IDs, words_mask, text_lengths, and input_ids between Forge and IO paths |
| Post-processing parity | PASS | Identical decoded entities (3 entities: CEO/person, Apple/org, Paris/location @ 0.8808) |

### Key Finding
The IOProcessor produces **bit-identical** preprocessing and postprocessing output compared to the existing Forge `BaseProcessor`. Both paths use the same `GLiNERPreprocessor` and `GLiNERDecoder` — the IOProcessor simply wraps them in vLLM's native interface.

## 2. HTTP/API Simplification

### What the IOProcessor path eliminates

| Component | Current (Forge) Path | IOProcessor Path | Eliminated? |
|-----------|---------------------|------------------|-------------|
| `forge/patches/pooling_extra_kwargs.py` — Request patch | Patches `PoolingCompletionRequest` to add `extra_kwargs` field | `IOProcessorRequest(data=...)` carries arbitrary JSON natively | **Yes** (request side) |
| `forge/patches/pooling_extra_kwargs.py` — Response patch | Patches `PoolingResponseData.data` from typed to `Any` | `IOProcessorResponse(data=...)` accepts arbitrary data natively | **Yes** (response side) |
| `Dockerfile RUN python -m forge.patches.pooling_extra_kwargs` | Required before server start | Not needed | **Yes** |
| `forge/processor_base.py` + per-plugin `processor.py` | Custom Python-side pipeline wrapper | Built into vLLM's serving layer | **Partially** — still needed for offline/testing but served path uses vLLM directly |
| Client `extra_kwargs` contract | `POST /v1/pooling {"input": "...", "extra_kwargs": {...}}` | `POST /pooling {"data": {"text": "...", "labels": [...]}}` | **Changed** — cleaner, schema-defined |

### What the IOProcessor path does NOT eliminate

| Component | Reason |
|-----------|--------|
| `PoolingParams.extra_kwargs` at engine level | The model/pooler still receives structured data via `extra_kwargs`; IOProcessor sets this server-side via `validate_or_generate_params()` |
| `GPUModelRunner._preprocess` attention mask patch (linker plugins only) | Orthogonal to I/O layer — deep engine constraint |
| Model registration (`vllm.general_plugins`) | IOProcessor plugins are a separate entry-point group; model registration still needed |

### API Contract Change

**Before (current):**
```json
POST /v1/pooling
{
  "model": "gliner-mmbert",
  "input": "The CEO of Apple visited Paris.",
  "extra_kwargs": {
    "entities": ["person", "organization", "location"]
  }
}
```

**After (IOProcessor):**
```json
POST /pooling
{
  "model": "gliner-mmbert",
  "task": "plugin",
  "data": {
    "text": "The CEO of Apple visited Paris.",
    "labels": ["person", "organization", "location"],
    "threshold": 0.5
  }
}
```

Benefits:
- `data` is schema-validated by `parse_request()` — structured errors instead of silent passthrough
- Response is `IOProcessorResponse(data=[...entities...])` — typed, not patched
- Client doesn't need to know about vLLM internals (`extra_kwargs`, `PoolingParams`)

## 3. Performance Implications

### No regression expected

The IOProcessor path introduces **zero additional computation**:
- Pre-processing: Same `GLiNERPreprocessor` call, same tokenization
- Engine: Same `engine_client.encode()` with same `PoolingParams(extra_kwargs=...)`
- Post-processing: Same `GLiNERDecoder.decode()` + `get_final_entities()`

The only difference is the call path:
- **Current:** Client → HTTP handler → Forge processor.py → engine.encode → Forge postprocess → HTTP response
- **IOProcessor:** Client → HTTP handler → vLLM `create_pooling()` → IOProcessor.pre_process → engine.encode → IOProcessor.post_process → IOProcessor.output_to_response → HTTP response

The IOProcessor path is marginally simpler because it eliminates the external Forge process_single() retry loop, asyncio.Lock engine lifecycle management, and the manual PoolingParams → HTTP response translation.

### Stash-and-pop pattern

The `validate_or_generate_params()` method in the IOProcessor ABC doesn't accept per-request context. The pilot uses a stash-and-pop pattern:
1. `pre_process()` stashes `extra_kwargs` and metadata
2. `validate_or_generate_params()` pops `extra_kwargs` into `PoolingParams`
3. `post_process()` retrieves metadata by `request_id`

This is safe in vLLM 0.15.1 because:
- The serving code calls `pre_process_async()` → `validate_or_generate_params()` sequentially with no `await` in between
- Post-process correlation uses `request_id` with a thread-safe dict

However, this is an undocumented behavioral dependency. If vLLM changes the calling order or adds concurrency between these calls, the pattern would break.

## 4. Applicability to Other Plugins

Based on the audit matrix, if this pilot succeeds at production scale:

| Plugin | Migration Effort | Expected Benefit |
|--------|-----------------|------------------|
| `deberta_gliner` | Low (same pattern) | Eliminates both patches |
| `mt5_gliner` | Low (same pattern + attention_mask) | Eliminates both patches |
| `deberta_gliner2` | Medium (schema preprocessing) | Eliminates both patches |
| `moderncolbert` | Low | Eliminates extra_kwargs patch only |
| `colqwen3` | Low | Eliminates extra_kwargs patch only |
| `deberta_gliner_linker` | **Blocked** | AM patch not addressed |
| `modernbert_gliner_rerank` | **Blocked** | AM patch not addressed |

## 5. Decision-Relevant Summary

**Does the IOProcessor provide a substantial upgrade?**

For the GLiNER-family plugins (4 of 11): **Yes.** It eliminates the need for the pooling protocol patch entirely, provides a cleaner client API contract with input validation, and introduces no performance regression. The pre/post processing logic is bit-identical.

For the ColBERT/multimodal plugins (3 of 11): **Marginal.** Cleaner API, but the existing path works fine and the patch is low-risk.

For the blocked plugins (2 of 11): **No impact.** The engine-level attention mask patch is orthogonal.

For the remaining plugins (2 of 11): **No benefit.** No processor to migrate.

## 6. Multimodal Expansion — ColQwen3

A second IOProcessor was implemented for the `colqwen3` plugin (Qwen3-VL + ColPali multi-vector embeddings) to validate that the pattern generalizes beyond GLiNER-style structured extraction.

### Results (all PASS)

| Test | Result |
|------|--------|
| Entry-point discovery | PASS |
| Class resolution | PASS |
| ABC contract | PASS |
| parse_request (text + image + validation) | PASS |
| Params contract (task=token_embed, extra_kwargs) | PASS |
| Pre-processing parity (text + image inputs) | PASS |
| Post-processing parity (1024-dim embedding) | PASS |
| Response contract | PASS |

### Key Observations

1. The IOProcessor pattern is **trivially portable** — ColQwen3's processor has 6 lines of logic vs mmbert_gliner's 100+, and both map cleanly to the same IOProcessor interface.
2. Multimodal inputs (image dicts) pass through `pre_process` to vLLM without modification — the IOProcessor doesn't interfere with vLLM's image processing pipeline.
3. The `task="token_embed"` pooling task is correctly set via `validate_or_generate_params()`.

## 7. Decision Summary

**Recommendation:** Integrate IOProcessor for the GLiNER family as a **parallel path** (keeping the Forge path for backward compatibility and offline use). This eliminates the pooling patch for served models, which is the highest-maintenance component in the repo.

### Decision: Integrate IOProcessor on vLLM 0.15.1 — no upgrade needed

| Criterion | Assessment |
|-----------|-----------|
| Available in 0.15.1? | **Yes** — ABC, protocol types, serving wiring, and offline LLM.encode() all present |
| Parity quality? | **Bit-identical** — same preprocessor/postprocessor, same tokens, same entities |
| HTTP simplification? | **Yes** — eliminates both halves of pooling_extra_kwargs.py patch for served models |
| Performance regression? | **None** — zero additional computation, same engine path |
| Generalizes beyond GLiNER? | **Yes** — validated on multimodal ColPali with identical success |
| Upgrade needed? | **No** — all required APIs are in 0.15.1 |

### Implementation Roadmap

1. **Phase 1 (immediate):** Ship `mmbert_gliner_io` and `colqwen3_io` as parallel IOProcessor paths alongside existing Forge processors. Models that specify `"io_processor_plugin": "mmbert_gliner_io"` in their HF config (or via `--io-processor-plugin` at serve time) use the new path; all others remain on the existing path.

2. **Phase 2 (short-term):** Add IOProcessors for `deberta_gliner`, `mt5_gliner`, `deberta_gliner2`, and `moderncolbert`. These follow the same pattern validated in the pilot.

3. **Phase 3 (medium-term):** Once all high-value plugins have IOProcessors, deprecate the `forge/patches/pooling_extra_kwargs.py` patch for served deployments. Keep it available for edge cases or older clients.

4. **Deferred:** `deberta_gliner_linker` and `modernbert_gliner_rerank` remain on the Forge path until the `GPUModelRunner._preprocess` attention mask issue is resolved upstream or via a different mechanism.

### Caveats

1. **Stash-and-pop pattern:** The `validate_or_generate_params()` limitation in the IOProcessor ABC requires a stash-and-pop pattern to transfer per-request `extra_kwargs`. This works reliably in 0.15.1 but is an undocumented behavioral dependency.

2. **Client contract change:** Switching from `extra_kwargs`-based to `data`-based payloads requires updating `VllmPoolingProvider` in `latency-index-3` and any other clients. This is a one-time migration.

3. **Two entry-point groups:** Plugins now register in both `vllm.general_plugins` (model registration) and `vllm.io_processor_plugins` (IO processor). These serve different purposes and both are needed.
