# IO Processor Plugin — Migration Audit Matrix

Audit of all 11 vllm-factory plugins against the vLLM 0.15.1 `IOProcessor` ABC
(`vllm.plugins.io_processors.interface.IOProcessor`).

## Background

vLLM 0.15.1 ships with a first-class `IOProcessor` plugin system that:

- Receives arbitrary JSON via `IOProcessorRequest(data=...)` on `/pooling`
- Calls `parse_request()` → `pre_process()` → engine.encode → `post_process()` → `output_to_response()`
- Returns arbitrary JSON via `IOProcessorResponse(data=...)`
- Is wired into **both** offline (`LLM.encode()`) and online (`OpenAIServingPooling.create_pooling()`) paths
- Uses a **separate** entry-point group: `vllm.io_processor_plugins` (not `vllm.general_plugins`)

This means the IOProcessor path bypasses the normal `PoolingCompletionRequest` /
`PoolingResponseData` types entirely — the two things we currently patch via
`forge/patches/pooling_extra_kwargs.py`.

## Plugin Matrix

| Plugin | Processor Type | extra_kwargs | Response Depth | AM Patch | Pooling Patch Needed | IOProcessor Fit | Notes |
|--------|---------------|-------------|---------------|----------|---------------------|----------------|-------|
| `embeddinggemma` | Custom sync (LLM.embed) | None | 1D float | No | No | **Low** | Simple dense embed; no pre/post complexity to offload |
| `moderncolbert` | BaseProcessor | Yes (is_query, sequence_length, attention_mask, input_ids) | 2D float | No | Yes (extra_kwargs) | **Medium** | Pre-tokenization + prefix insertion could move to IOProcessor |
| `colqwen3` | BaseProcessor | Yes (is_query) | 2D float | No | Yes (extra_kwargs) | **Medium** | Thin pre/post; main gain is removing extra_kwargs patch dependency |
| `collfm2` | BaseProcessor | None | 2D float | No | No (task=token_embed) | **Low** | Multimodal prompt construction already works; no patch dependency |
| `nemotron_colembed` | None (serve-only) | — | 2D float | No | — | **Low** | No processor to migrate |
| `lfm2_colbert` | None (serve-only) | — | 2D float | No | — | **Low** | No processor to migrate |
| `mmbert_gliner` | BaseProcessor (GLiNER) | Yes (input_ids, words_mask, text_lengths) | 3D+ tensor | No | Yes (extra_kwargs + response Any) | **High** | Heavy pre/post; IOProcessor eliminates both patches |
| `deberta_gliner` | BaseProcessor (GLiNER) | Yes (input_ids, words_mask, text_lengths, attention_mask) | 3D+ tensor | No | Yes (extra_kwargs + response Any) | **High** | Same as mmbert_gliner; strong pilot candidate |
| `mt5_gliner` | BaseProcessor (GLiNER) | Yes (input_ids, words_mask, text_lengths, attention_mask) | 3D+ tensor | No | Yes (extra_kwargs + response Any) | **High** | Same pattern as deberta_gliner |
| `deberta_gliner2` | Custom (schema) | Yes (via extra_kwargs) | JSON-encoded bytes | No | Yes (extra_kwargs + response Any) | **High** | Complex schema pre/post; IOProcessor is a natural fit |
| `deberta_gliner_linker` | Custom (GLiNER L3) | Yes (input_ids, AM, words_mask, text_lengths, labels_embeds) | 1D packed tensor | **Yes** | Yes (extra_kwargs) | **Blocked** | Requires GPUModelRunner._preprocess AM patch; IOProcessor cannot help with that |
| `modernbert_gliner_rerank` | Custom (GLiNER L4) | Yes (input_ids, AM, words_mask, text_lengths) | 1D packed tensor | **Yes** | Yes (extra_kwargs) | **Blocked** | Same AM patch dependency; also has GPU instability issues |

## Key Findings

1. **No vLLM upgrade required.** IOProcessor ABC, protocol types, serving wiring, and offline LLM.encode() integration are all present in `0.15.1`.

2. **GLiNER-family plugins (mmbert, deberta, mt5, gliner2) are the highest-value targets.** They depend on both halves of the pooling patch (extra_kwargs passthrough AND 3D+ response type). Moving them to IOProcessor would eliminate the need for the patch entirely for those models.

3. **The two linker/rerank plugins remain blocked** regardless of IOProcessor, because their `GPUModelRunner._preprocess` monkey-patch for attention masks is orthogonal to the I/O layer.

4. **ColBERT/ColPali plugins benefit marginally.** They use extra_kwargs for lightweight metadata (is_query) but their responses are standard 2D float arrays. IOProcessor would clean up the contract but isn't a parity-critical improvement.

5. **embeddinggemma, nemotron_colembed, lfm2_colbert** have no meaningful pre/post processing complexity or patch dependency. Migration would be overhead with no benefit.

## Pilot Recommendation

**Start with `mmbert_gliner`** (ModernBERT + GLiNER span head):
- Uses shared `GLiNERPreprocessor` + `GLiNERDecoder` — representative of the entire GLiNER family
- No attention-mask patch dependency (unlike linker plugins)
- Already has proven F1=1.0000 parity against reference GLiNER
- Exercises both patch components (extra_kwargs passthrough + 3D response)
- If IOProcessor achieves parity here, the same pattern applies to deberta_gliner, mt5_gliner, and deberta_gliner2

## Client Impact

- `latency-index-3/voyager_index/multimodal.py` (`VllmPoolingProvider`) currently sends to `/v1/pooling` with `extra_kwargs`. IOProcessor requests go to `/pooling` with `data: {...}` instead. The client contract would change from `extra_kwargs`-based to `data`-based payloads.
- Offline users calling `LLM.encode(prompt_dict_with_data_key)` get automatic IOProcessor routing.
