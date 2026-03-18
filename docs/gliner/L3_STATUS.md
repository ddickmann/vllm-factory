# vLLM Integration Status — GLiNER Linker Plugin (L3)

> **Production-ready.** Entity parity uses recall-gated validation: every reference entity must be found by vLLM. Score deltas are reported but not gating (bfloat16 naturally produces small drift vs fp32 reference). See [`README.md`](README.md) and [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md).

**Date**: 2026-03-19  
**Plugin**: `deberta_gliner_linker`  
**Model**: [`knowledgator/gliner-linker-large-v1.0`](https://huggingface.co/knowledgator/gliner-linker-large-v1.0) (BiEncoderTokenModel, DeBERTa-v3-large)

**Handover / architecture / patches:** [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) (L3 + L4).

## GLiNKER / Hugging Face reference

- **L3 linker** (this plugin): `gliner-linker-{base,large}-v1.0` — same DeBERTa family for text + label encoders per model card.  
- **L4 reranker**: `gliner-linker-rerank-v1.0` — **ettin-encoder-68m**; vLLM plugin [`modernbert_gliner_rerank`](../../plugins/modernbert_gliner_rerank/). Docs: [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md), [`OVERVIEW.md`](OVERVIEW.md). See also [`L4_NOTES.md`](L4_NOTES.md).  
- Install pattern: `pip install git+https://github.com/Knowledgator/GLinker.git` (pulls `gliner`).

## What works (proven)

### 1. Logits parity: `parity_test.py` — cos_sim ≈ 1.0

Text-only (regex words, no ENT/label prompt) reference matches vLLM on the **HF-equivalent** stack: text encoder → **word gather** → scorer (**no** LSTM — matches `gliner.modeling.base.BiEncoderTokenModel.forward`).

**Run**: `cd vllm-factory && PYTHONPATH=. python scripts/gliner/l3/parity_test.py`

### 2. Preprocess parity: `preprocess_parity_test.py`

`BiEncoderTokenDataCollator` output matches `GLiNERLinkerProcessor._tokenize` (input_ids, words_mask, text_lengths, words). Skips vLLM (`_ensure_llm` monkeypatch).

**Run**: `cd vllm-factory && PYTHONPATH=. python scripts/gliner/l3/preprocess_parity_test.py`

### 3. L3 entity parity: `entity_parity_test.py`

Phase 1: GLinker `ProcessorFactory` (native PyTorch). Phase 2: `GLiNERLinkerProcessor` + vLLM. Subprocesses avoid GPU fork issues.

- **Structural**: same entity count; each entity matches **text**, **start**, **end**, **label** (sorted).  
- **Scores**: label embeddings must match GLinker precompute: cap `labels_tokenizer.model_max_length` to the same value as `L3Config.max_length` (512) before `encode_labels`, and batch like GLinker (`batch_size` 32). Otherwise HF `padding="max_length"` is a no-op and `batch_size=1` label encoding diverges — that alone produced ~0.3 score gaps. Remaining drift is vLLM vs HF on the **text** encoder; default absolute tolerance **0.05** (`GLINKER_ENTITY_SCORE_ATOL`).

**Run**: `cd vllm-factory && PYTHONPATH=. python scripts/gliner/l3/entity_parity_test.py [--regen]`

### 4. Implementation notes

| Topic | Detail |
|--------|--------|
| **ENT/SEP prompt** | GLiNER prepends `[ENT, label, …, SEP]` before text words; the processor uses `BiEncoderTokenDataCollator` + `entity_types=labels` — not raw words only. |
| **LSTM in checkpoint** | Weights exist under `rnn.lstm.*` but **GLiNER does not run LSTM** before the scorer on this model; the pooler matches that. |
| **Span.end** | TokenDecoder uses **inclusive** end word index; char spans use `word_ends[span.end]`. |
| **Attention mask** | Collator `attention_mask` is checked in `preprocess_parity_test.py`, sent in `PoolingParams.extra_kwargs`, and **injected into `model.forward`** via a one-time monkey-patch of vLLM v1 `GPUModelRunner._preprocess` (`vllm_pooling_attention_mask.py`). Masks are flattened in scheduler order, sliced for chunked prefill (`[num_computed:num_computed+L]`), and zero-padded to vLLM’s token buffer width. `GLiNERLinkerModel.forward` applies them when building batched DeBERTa inputs. Unit checks: `attention_mask_concat_test.py`. |
| **Decode threshold** | Same as GLiNER: `TokenDecoder.decode(..., threshold=threshold, flat_ner=..., multi_label=...)`. Using `threshold=0.0` inside decode is wrong because `sigmoid > 0` is always true, which changes start/end candidates and `greedy_search`. |
| **Entity `text` field** | Same as GLiNER: substring `source_text[start:end]` with **exclusive** `end` from `words_splitter` (not `" ".join(words[...])`, which can diverge on whitespace). |
| **vLLM serve** | `vllm serve knowledgator/gliner-linker-large-v1.0 --trust-remote-code --dtype float32 --enforce-eager` (with prepared cache from `get_model_path()` as in plugin). |

## Architecture files

| Location | Role |
|----------|------|
| [`plugins/deberta_gliner_linker/model.py`](../../plugins/deberta_gliner_linker/model.py) | vLLM text (+ labels) DeBERTa wrapper |
| [`plugins/deberta_gliner_linker/pooler.py`](../../plugins/deberta_gliner_linker/pooler.py) | words_mask gather + scorer → flat (W,C,3) embedding |
| [`plugins/deberta_gliner_linker/processor.py`](../../plugins/deberta_gliner_linker/processor.py) | Collator-aligned tensors + `LLM.embed` + `TokenDecoder` |
| [`scripts/gliner/l3/parity_test.py`](../../scripts/gliner/l3/parity_test.py) | HF vs vLLM logits (text-only recipe) |
| [`scripts/gliner/l3/preprocess_parity_test.py`](../../scripts/gliner/l3/preprocess_parity_test.py) | Collator vs `_tokenize` tensors |
| [`scripts/gliner/l3/entity_parity_test.py`](../../scripts/gliner/l3/entity_parity_test.py) | GLinker L3 vs processor end-to-end |
| [`plugins/deberta_gliner_linker/__init__.py`](../../plugins/deberta_gliner_linker/__init__.py) | Plugin registration + local model cache |

## Optional dependencies

Install for tests: `pip install vllm-factory[glinker]` (see root `pyproject.toml`).

## Historical notes (fixed)

Earlier issues: (1) regex tokenization without ENT/label prompt; (2) wrong `Span.end` interpretation; (3) pooler applied LSTM though GLiNER forward does not; (4) `encode_labels` without capping the labels tokenizer’s `model_max_length` (GLinker does this in `L3Component._setup`) — all addressed in the current tree.
