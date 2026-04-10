# DeBERTa-GLiNER2

Schema-driven entity extraction, classification, relations via GLiNER2.

**Model:** [fastino/gliner2-large-v1](https://huggingface.co/fastino/gliner2-large-v1)
**Architecture:** DeBERTa v3-large backbone, GLiNER2 schema processor (entity/classification/relation/JSON)
**Performance:** Speedup vs vanilla GLiNER2, scales with batch size — run `benchmark_gliner2.py` on your hardware
**Parity:** Entity F1 = 1.0000, Classification ✅, Relations ✅, JSON ✅

## Usage

```python
from vllm import LLM

llm = LLM("fastino/gliner2-large-v1", trust_remote_code=True)
# GLiNER2 supports simple labels, schema requests, and mixed schema output
```

## Request Contract

- `labels` stays supported for the simple entity-extraction path.
- `schema` supports mixed-task requests.
- If both are present, `schema` wins.
- One request can mix `entities`, `classifications`, `relations`, and `structures`.
- `threshold`, `include_confidence`, and `include_spans` are request-level flags.
- `labels` is the simplest path for benchmarks and parity tests.

## Serve

Requires a prepared model directory (see `forge/model_prep.py`).

```bash
vllm serve /tmp/gliner2-vllm \
  --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --port 8200
```

## Verify

```bash
python plugins/deberta_gliner2/parity_test.py
```

For the parser/runtime contract tests:

```bash
python -m pytest tests/test_gliner2_io_processor_contract.py tests/test_gliner2_schema_contract.py tests/test_gliner2_runtime_contract.py
```
