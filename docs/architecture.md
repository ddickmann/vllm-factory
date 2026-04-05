# Architecture

## Overview (0.2.0)

vllm-factory uses three abstraction layers to decouple plugin logic from vLLM internals:

| Layer | File | Role | vLLM imports? |
|---|---|---|---|
| **FactoryIOProcessor** | `vllm_factory/io/base.py` | Pre/post-processing adapter (subclass of vLLM's `IOProcessor`) | Yes (constructor only) |
| **FactoryPooler** | `vllm_factory/pooling/protocol.py` | Business logic protocol — task-specific tensor transforms | **No** |
| **VllmPoolerAdapter** | `vllm_factory/pooling/vllm_adapter.py` | Bridges `FactoryPooler` to vLLM's Pooler ABC | Yes (single file) |

Plugin authors implement `FactoryIOProcessor` and `FactoryPooler`. The adapter layer is shared and maintained once.

## Data Flow

```
POST /pooling {"data": {"text": "...", "labels": [...]}}
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  FactoryIOProcessor (plugin-specific)                            │
│    parse_request()        → extract text, labels, params         │
│    factory_pre_process()  → tokenize, build extra_kwargs         │
│    PoolingParams(extra_kwargs={...})                              │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  vLLM V1 Engine                                                  │
│    scheduler → batches requests                                  │
│    model.forward(input_ids, positions, **kwargs) → hidden_states │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  VllmPoolerAdapter (shared)                                      │
│    extracts PoolerContext from vLLM's PoolingMetadata             │
│    calls FactoryPooler.forward(hidden_states, context)           │
│    returns list[PoolingSequenceGroupOutput]                      │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  FactoryIOProcessor (plugin-specific)                            │
│    factory_post_process() → decode entities / format embeddings  │
│    output_to_response()   → JSON response                       │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
{"data": [{"text": "Apple Inc.", "label": "company", "score": 0.95}]}
```

## How Plugins Load

```
pyproject.toml entry points:
  vllm.general_plugins → plugin.__init__:register()
  vllm.io_processor_plugins → plugin.io_processor:get_processor_cls()
    │
    ▼
register() calls:
  AutoConfig.register("model_type", PluginConfig)
  ModelRegistry.register_model("ModelName", PluginModel)
    │
    ▼
vllm serve model-name --runner pooling --io-processor-plugin plugin_io
    │
    ▼
vLLM reads config.json → model_type → finds PluginConfig
    → instantiates PluginModel(vllm_config=...)
    → loads weights
    → serves with FactoryIOProcessor handling all I/O
```

## Plugin Structure

```
Plugin = Config + Model + FactoryPooler + FactoryIOProcessor
```

| Component | Role | Required |
|---|---|---|
| **Config** (`config.py`) | Tells vLLM model dimensions, layer count, vocab size | Yes |
| **Model** (`model.py`) | Runs the encoder, returns hidden states | Yes |
| **Pooler** (`pooler.py`) | Implements `FactoryPooler` — transforms hidden states → task output | Yes |
| **IOProcessor** (`io_processor.py`) | Subclasses `FactoryIOProcessor` — server-side pre/post-processing | Yes |

## FactoryPooler Protocol

```python
class FactoryPooler(Protocol):
    def forward(
        self,
        hidden_states: torch.Tensor,
        context: PoolerContext,
    ) -> list[torch.Tensor]:
        ...
```

`PoolerContext` is a stable internal data model (`vllm_factory/pooling/context.py`) containing `prompt_token_ids`, `extra_kwargs`, and batch metadata. No vLLM types leak into pooler logic.

## Pooler Organization

### Shared Poolers (`poolers/`)

Reusable across multiple plugins:

| File | Used By | FactoryPooler? |
|---|---|---|
| `colbert.py` | moderncolbert | Yes |
| `gliner.py` | deberta_gliner, mmbert_gliner, mt5_gliner | Yes |
| `gliner2.py` | deberta_gliner2 | Yes |
| `colpali.py` | colqwen3, collfm2 | Yes |

### Plugin-Specific Poolers (`plugins/*/pooler.py`)

| Plugin | Why Specific |
|---|---|
| `deberta_gliner_linker` | Word gather + labels encoder + scorer (bi-encoder path) |
| `modernbert_gliner_rerank` | Prompt/word extract + LSTM + scorer (uni-encoder path) |
| `embeddinggemma` | Custom CLS + learned projection head |
| `lfm2_colbert` | Uses vLLM's built-in tokwise pooler |
| `nemotron_colembed` | PassthroughPooler (token embedding via vLLM native path) |

## Weight Loading

```
HuggingFace checkpoint (safetensors/bin)
    → vLLM weight iterator yields (name, tensor) pairs
    → model.load_weights() maps checkpoint keys → model parameter names
    → param.weight_loader(param, loaded_weight) copies data
```

## vLLM Decorators

| Decorator | Purpose |
|---|---|
| `@attn_type("encoder_only")` | Bidirectional attention (not causal) |
| `@default_pooling_type(tok_pooling_type="ALL")` | Return all token embeddings to pooler |
| `is_pooling_model = True` | Use pooling runner, not generation |

## Runtime Monkey-Patches

Two patches remain in 0.2.0. Both are idempotent, scoped to model `__init__`, and documented in-code with "WHY / CHARACTERISTICS / UPSTREAM RESOLUTION" sections.

| Patch | File | Applied In | Removes When |
|---|---|---|---|
| `_preprocess` attention_mask | `plugins/deberta_gliner_linker/vllm_pooling_attention_mask.py` | `GLiNERLinkerModel.__init__` | vLLM forwards all `extra_kwargs` into `model_kwargs` |
| `get_kv_cache_spec` skip | `plugins/nemotron_colembed/model.py` | `NemotronColEmbedModel.__init__` | vLLM returns `None` for `ENCODER_ONLY` layers |
