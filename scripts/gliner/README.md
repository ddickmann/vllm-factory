# GLiNER / Knowledgator parity & ad-hoc scripts

> Development and parity tooling for GLiNER plugins. See **[docs/gliner/README.md](../../docs/gliner/README.md)**.

Run from repo root with **`PYTHONPATH=.`** (or `cd vllm-factory`).

## L3 linker (`deberta_gliner_linker`)

| Script | Purpose |
|--------|---------|
| [`l3/parity_test.py`](l3/parity_test.py) | HF vs vLLM logits (text-only recipe) |
| [`l3/preprocess_parity_test.py`](l3/preprocess_parity_test.py) | Collator vs `GLiNERLinkerProcessor._tokenize` |
| [`l3/entity_parity_test.py`](l3/entity_parity_test.py) | End-to-end entities vs native GLinker |
| [`l3/attention_mask_concat_test.py`](l3/attention_mask_concat_test.py) | Flat attention-mask construction |
| [`l3/adhoc_test_labels_v2.py`](l3/adhoc_test_labels_v2.py) | Ad-hoc label / vLLM init experiment |
| [`l3/adhoc_entity_vs_generic_labels.py`](l3/adhoc_entity_vs_generic_labels.py) | Ad-hoc entity vs generic label encoding |

```bash
cd vllm-factory && PYTHONPATH=. python scripts/gliner/l3/parity_test.py
```

## L4 rerank (`modernbert_gliner_rerank`)

| Script | Purpose |
|--------|---------|
| [`l4/l4_parity_fixtures.py`](l4/l4_parity_fixtures.py) | Shared text/label fixtures (imported by scripts + pytest) |
| [`l4/preprocess_parity_test.py`](l4/preprocess_parity_test.py) | Collator vs `GLiNERRerankProcessor._tokenize` |
| [`l4/entity_parity_test.py`](l4/entity_parity_test.py) | Native GLiNER vs vLLM processor (GPU) |
| [`l4/batch_vllm_parity_test.py`](l4/batch_vllm_parity_test.py) | Batched vs sequential `embed` (GPU) |

```bash
cd vllm-factory && PYTHONPATH=. python scripts/gliner/l4/preprocess_parity_test.py
```

**Docs:** [`docs/gliner/README.md`](../../docs/gliner/README.md)
