# GLinker / GLiNER linker plugins for vLLM Factory

> **Production-ready.** All GLiNER plugins pass end-to-end parity testing. See [`README.md`](README.md).

Short overview. For **handover depth** (patches, quirks, why they exist, usage, verification), use **[`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md)**.

---

## Model → plugin map

| Stage | HF model | Plugin package | Entry point (`pyproject.toml`) |
|--------|-----------|----------------|------------------------------|
| L3 linker | [`knowledgator/gliner-linker-large-v1.0`](https://huggingface.co/knowledgator/gliner-linker-large-v1.0) | [`plugins/deberta_gliner_linker`](../../plugins/deberta_gliner_linker/) | `deberta_gliner_linker` |
| L4 rerank | [`knowledgator/gliner-linker-rerank-v1.0`](https://huggingface.co/knowledgator/gliner-linker-rerank-v1.0) | [`plugins/modernbert_gliner_rerank`](../../plugins/modernbert_gliner_rerank/) | `modernbert_gliner_rerank` |

Both are **pooling models**: use **`LLM.embed`** (or OpenAI-compatible pooling), not text generation. Custom poolers emit flattened logits; Python **processors** run **`TokenDecoder`**.

---

## Prerequisites

- **`pip install -e .`** from repo root (entry points + plugins).
- **`gliner`**: `pip install "vllm-factory[glinker]"` or equivalent.
- **GPU** for real engines.
- Plugins load via **`vllm.general_plugins`**:

  ```bash
  export VLLM_PLUGINS=modernbert_gliner_rerank,deberta_gliner_linker
  ```

---

## One-line “what is hacked”

vLLM v1 does not pass **`PoolingParams.extra_kwargs["attention_mask"]`** into **`model.forward`**. Both plugins register a **monkey-patch** on **`GPUModelRunner._preprocess`** ([`vllm_pooling_attention_mask.py`](../../plugins/deberta_gliner_linker/vllm_pooling_attention_mask.py)) that builds a **flat** mask across scheduled requests. **Why / edge cases / chunked prefill:** see [`INTEGRATION_GUIDE.md` §3](INTEGRATION_GUIDE.md#3-shared-hack-pooling-attention_mask-monkey-patch).

---

## Get a local model directory

**L3:**

```python
from plugins.deberta_gliner_linker import get_model_path
model_dir = get_model_path()
```

**L4:**

```python
from plugins.modernbert_gliner_rerank import get_model_path
model_dir = get_model_path()
```

---

## Processors (typical app usage)

**L3:** [`GLiNERLinkerProcessor`](../../plugins/deberta_gliner_linker/processor.py) — `warmup(labels)` → `predict_entities` / `batch_predict_entities` → `close()`.

**L4:** [`GLiNERRerankProcessor`](../../plugins/modernbert_gliner_rerank/processor.py) — same surface API; **no** `encode_labels` in warmup.

Full patterns and **`extra_kwargs` keys:** [`INTEGRATION_GUIDE.md` §7](INTEGRATION_GUIDE.md#7-usage-patterns).

---

## Parity & tests (quick index)

Scripts live under **`scripts/gliner/`** — see [`scripts/gliner/README.md`](../../scripts/gliner/README.md).

| Plugin | CPU | GPU / manual |
|--------|-----|----------------|
| L3 | [`scripts/gliner/l3/preprocess_parity_test.py`](../../scripts/gliner/l3/preprocess_parity_test.py) | [`scripts/gliner/l3/parity_test.py`](../../scripts/gliner/l3/parity_test.py), [`scripts/gliner/l3/entity_parity_test.py`](../../scripts/gliner/l3/entity_parity_test.py) |
| L4 | [`scripts/gliner/l4/preprocess_parity_test.py`](../../scripts/gliner/l4/preprocess_parity_test.py), `pytest tests/test_modernbert_gliner_rerank_batch_contract.py` | [`scripts/gliner/l4/batch_vllm_parity_test.py`](../../scripts/gliner/l4/batch_vllm_parity_test.py), [`scripts/gliner/l4/entity_parity_test.py`](../../scripts/gliner/l4/entity_parity_test.py) (often blocked — see [`L4_PARITY.md`](L4_PARITY.md)) |

L3 status notes: [`L3_STATUS.md`](L3_STATUS.md).

---

## Serve example

```bash
export VLLM_PLUGINS=deberta_gliner_linker
vllm serve "$(python -c 'from plugins.deberta_gliner_linker import get_model_path; print(get_model_path())')" \
  --trust-remote-code --runner pooling
```

(Swap plugin + `get_model_path` import for L4.) CLI details vary by vLLM version.

---

## Choosing L3 vs L4

| | L3 linker | L4 rerank |
|---|-----------|-----------|
| GLiNER class | `BiEncoderTokenModel` | `UniEncoderTokenModel` + LSTM |
| Text encoder | DeBERTa v1 (custom) | ModernBERT (ettin) |
| Labels | Second encoder or `labels_embeds` | Prompt in one sequence |
| Collator | `BiEncoderTokenDataCollator` | `TokenDataCollator` |

---

## Troubleshooting

1. **`config.json` missing** — Call `get_model_path()` for the correct plugin.
2. **`attention_mask` length errors** — Must equal `num_prompt_tokens` per request (patch enforces).
3. **L3 score drift** — Label tokenizer `model_max_length` + batched `encode_labels` (see integration guide §4.3).
4. **Plugin not registered** — `VLLM_PLUGINS` + editable install.
5. **L4 GPU crashes** — Known vLLM + ModernBERT attention issues; see [`L4_PARITY.md`](L4_PARITY.md).

---

## Related files

- **[`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md)** — full handover
- [`L3_STATUS.md`](L3_STATUS.md), [`L4_NOTES.md`](L4_NOTES.md), [`L4_PARITY.md`](L4_PARITY.md)
- [`forge/validate_plugins.py`](../../forge/validate_plugins.py)
