# Architecture

## How Plugins Load

```
pip install -e plugins/my_model/
    → setup.py entry_points registers vllm.general_plugins
    → vLLM calls register() on import
    → ModelRegistry.register_model("MyModel", MyModelClass)
    → AutoConfig.register("my_model", MyConfig)
    → vLLM can now load your model via LLM(model="...")
```

## Plugin Structure

```
Plugin = Config + Model + Pooler + Processor
```

| Component | Role | Required |
|---|---|---|
| **Config** | Tells vLLM model dimensions, layer count, vocab size | Yes |
| **Model** | Runs the encoder, returns hidden states | Yes |
| **Pooler** | Transforms hidden states → task-specific output | Yes |
| **Processor** | Pre/post-processing for the HTTP API | No |

## Data Flow

```
Input text
    → vLLM tokenizer
    → model.forward(input_ids, positions)  →  hidden_states (total_tokens, H)
    → pooler.forward(hidden_states, pooling_metadata)  →  list[tensor]
    → client receives embedding tensor
```

For models with `extra_kwargs` (GLiNER, Linker):
```
Client sends: {"input": "...", "extra_kwargs": {"words_mask": [...], "labels": [...]}}
    → PoolingParams.extra_kwargs passes through to pooler
    → pooler reads metadata from pooling_metadata.pooling_params
```

## Pooler Organization

Two types:

### Shared Poolers (`poolers/`)

Reusable across multiple plugins:

| File | Used By |
|---|---|
| `colbert.py` | moderncolbert |
| `gliner.py` | deberta_gliner, mmbert_gliner, mt5_gliner |
| `gliner2.py` | deberta_gliner2 |
| `colpali.py` | colqwen3, collfm2 |

### Plugin-Specific Poolers (`plugins/*/pooler.py`)

Tightly coupled to their model:

| Plugin | Why Specific |
|---|---|
| `deberta_gliner_linker` | Word gather + labels encoder + scorer (bi-encoder path) |
| `modernbert_gliner_rerank` | Prompt/word extract + LSTM + scorer (uni-encoder path) |
| `embeddinggemma` | Custom CLS + learned projection head |
| `lfm2_colbert` | Uses vLLM's built-in tokwise pooler |

**Rule of thumb**: If multiple plugins share the same pooler logic → `poolers/`. If the pooler needs model-specific components → `plugins/*/pooler.py`.

## Weight Loading

```
HuggingFace checkpoint (safetensors/bin)
    → vLLM weight iterator yields (name, tensor) pairs
    → model.load_weights() maps checkpoint keys → model parameter names
    → param.weight_loader(param, loaded_weight) copies data
```

Key mapping example (DeBERTa GLiNER):
```python
# Checkpoint: "deberta.embeddings.word_embeddings.weight"
# Model:      "model.embeddings.word_embeddings.weight"
# Mapping:    {"deberta.": "model."}
```

## vLLM Decorators

| Decorator | Purpose |
|---|---|
| `@attn_type("encoder_only")` | Bidirectional attention (not causal) |
| `@default_pooling_type(tok_pooling_type="ALL")` | Return all token embeddings to pooler |
| `is_pooling_model = True` | Use pooling runner, not generation |
