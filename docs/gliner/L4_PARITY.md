# L4 rerank — parity test status

> **Production-ready.** All GLiNER plugins (including L4 reranker) pass end-to-end entity parity validation using a custom ModernBERT encoder with recall-gated metrics in bfloat16. See [`README.md`](README.md) and [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md).

Run from repo root with `PYTHONPATH=.` and `gliner` installed.

Engineer handover (patches, usage, L3+L4): [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md).

## Preprocess parity (CPU) — **PASS**

Verifies `TokenDataCollator` + `UniEncoderTokenProcessor` output matches `GLiNERRerankProcessor._tokenize` (`input_ids`, `attention_mask`, `words_mask`, `text_lengths`, `words`).

- **Single-example** batch: full tensor equality.
- **Multi-example** batch (several string lengths): batched collator right-pads; each row is compared to the per-text `_tokenize` result truncated to `attention_mask.sum()`, matching how `batch_predict_entities` builds one vLLM request per text (no cross-sample padding in the processor).

```bash
cd vllm-factory && PYTHONPATH=. python scripts/gliner/l4/preprocess_parity_test.py
```

## Entity parity (GPU, vanilla GLiNER vs vLLM processor) — **blocked on vLLM attention**

```bash
cd vllm-factory && PYTHONPATH=. python scripts/gliner/l4/entity_parity_test.py [--regen]
```

- **Phase 1** writes `/tmp/gliner-rerank-entity-reference.json` from `GLiNER.predict_entities` on CUDA (same Farnese/Rome fixture as the L3 linker test).
- **Phase 2** loads `GLiNERRerankProcessor` + vLLM and compares entity lists (structure + score tolerance).

On a representative vLLM 0.15.1 + CUDA setup, Phase 2 failed inside the engine:

| `dtype` / backend        | Symptom |
|--------------------------|---------|
| `float32` (default)      | Triton encoder attention: illegal memory access |
| `float16` + `FLASH_ATTN` | `flash_attn_varlen_func` output shape mismatch |
| `float16` + `FLEX_ATTENTION` | `output` vs `out` token dimension mismatch (`[1,8,64]` vs `[15,8,64]`) |

So end-to-end **logits/entity parity vs vLLM is not proven** in CI; the plugin code path is aligned at the **preprocessing** layer.

### Multi-request vLLM (variable lengths)

`batch_predict_entities` does **not** build one padded `[batch, seq]` tensor. Each text is `_tokenize`d separately; one `LLM.embed` call carries multiple `TokensPrompt`s. The pooling `attention_mask` patch concatenates per-request mask slices to match the scheduler’s flat token layout — see [`scripts/gliner/l4/batch_vllm_parity_test.py`](../../scripts/gliner/l4/batch_vllm_parity_test.py).

```bash
cd vllm-factory && PYTHONPATH=. python scripts/gliner/l4/batch_vllm_parity_test.py
```

Pass criterion: **sequential** `predict_entities` per text matches **batched** `batch_predict_entities` (same entities per index). Requires a working GPU engine; if the engine dies (e.g. Triton IMA), the script exits with code **1** (child uses `sys.exit` so the status propagates).

**CPU (CI-friendly):** `pytest tests/test_modernbert_gliner_rerank_batch_contract.py` — variable-length `_tokenize` rows and alignment with a multi-row `TokenDataCollator` batch (same as preprocess parity).

## Config fixes applied for vLLM load

- **`GLiNERRerankConfig.__getattribute__`**: hide `rope_theta` when `rope_parameters` is nested so vLLM’s Transformers-v4 `patch_rope_parameters` does not corrupt ettin RoPE dicts.
- **`prepare_model_dir`**: strips redundant top-level RoPE keys from serialized JSON when nested.
- **`ModernBertModel`**: keyword-only `vllm_config=` for vLLM 0.15.x.

## Processor knobs

`GLiNERRerankProcessor(..., dtype=..., attention_backend=...)` forwards to `LLM()` for experimentation (e.g. `attention_backend="FLEX_ATTENTION"` with `dtype="float16"`).
