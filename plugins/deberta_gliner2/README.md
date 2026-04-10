# DeBERTa-GLiNER2

Schema-driven GLiNER2 extraction on top of the standard vLLM `POST /pooling` endpoint.

**Model:** [fastino/gliner2-large-v1](https://huggingface.co/fastino/gliner2-large-v1)  
**Tasks:** entities, classifications, relations, structures  
**Runtime:** native vLLM batching/scheduling, no separate FastAPI adapter

## Canonical Request

```json
{
  "model": "/models/gliner2-vllm",
  "data": {
    "text": "Tim Cook announced the iPhone 15 at Apple Park.",
    "schema": {
      "entities": {
        "person": "Person names",
        "company": "Company names",
        "product": "Product names",
        "location": "Places and locations"
      },
      "classifications": [
        {
          "task": "sentiment",
          "labels": ["positive", "negative", "neutral"]
        }
      ],
      "relations": ["works_for", "located_in"],
      "structures": {
        "product_summary": {
          "fields": [
            {"name": "name", "dtype": "str"},
            {"name": "launch_site", "dtype": "str"},
            {"name": "highlights", "dtype": "list"}
          ]
        }
      }
    },
    "threshold": 0.5,
    "include_confidence": true,
    "include_spans": true
  }
}
```

## Supported Schema Sections

- `entities`: `List[str]` or `Dict[str, str]`
- `classifications`: `List[{"task": ..., "labels": [...]}]`
- `relations`: `List[str]` or `Dict[str, str]`
- `structures`: `Dict[str, {"fields": List[field_def]}]`

## Examples

Entity extraction:
```json
{
  "data": {
    "text": "Apple released iPhone 15 in Cupertino.",
    "schema": {
      "entities": {
        "company": "Company names",
        "product": "Product names",
        "location": "Places and locations"
      }
    }
  }
}
```

Classification:
```json
{
  "data": {
    "text": "This support request is urgent and billing related.",
    "schema": {
      "classifications": [
        {
          "task": "priority",
          "labels": ["urgent", "high", "normal", "low"]
        },
        {
          "task": "department",
          "labels": ["sales", "support", "billing", "other"]
        }
      ]
    },
    "include_confidence": true
  }
}
```

Relations:
```json
{
  "data": {
    "text": "John Smith works at NVIDIA and reports to Jensen Huang.",
    "schema": {
      "relations": {
        "works_for": "Employment relationship",
        "reports_to": "Manager relationship"
      }
    },
    "include_confidence": true
  }
}
```

Structures:
```json
{
  "data": {
    "text": "The MacBook Pro costs $1999 and includes an M3 chip, 16GB RAM, and 512GB storage.",
    "schema": {
      "structures": {
        "product": {
          "fields": [
            {"name": "name", "dtype": "str", "description": "Product name"},
            {"name": "price", "dtype": "str", "description": "Listed price"},
            {"name": "features", "dtype": "list", "description": "Key features"}
          ]
        }
      }
    },
    "include_confidence": true
  }
}
```

## Notes

- `data.schema` is the only supported contract.
- One request supports mixed schemas in the same call.
- `threshold` defaults to `0.5`.
- `include_confidence` defaults to `false`.
- `include_spans` defaults to `false`.

## Not Supported Yet

- multiple texts in a single `/pooling` request

## Serve

Requires a prepared model directory.

```bash
vllm serve /tmp/gliner2-vllm \
  --runner pooling \
  --io-processor-plugin deberta_gliner2_io \
  --trust-remote-code --dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --port 8200
```

## Verify

```bash
uv run --with pytest python -m pytest tests/test_gliner2_schema_contract.py
uv run --with pytest python -m pytest tests/test_gliner2_io_processor_contract.py
uv run --with pytest python -m pytest tests/test_gliner2_runtime_contract.py
```
