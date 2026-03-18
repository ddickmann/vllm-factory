# Building a Plugin

A plugin is a pip-installable package that registers a custom model + pooler with vLLM.

## What's in a Plugin

```
plugins/my_model/
├── __init__.py      # Register model + config with vLLM
├── config.py        # HuggingFace-compatible config class
├── model.py         # Encoder model + pooler wiring
├── pooler.py        # (optional) Custom pooler head
├── processor.py     # (optional) Pre/post-processing pipeline
├── parity_test.py   # Parity test vs reference implementation
├── benchmark.py     # (optional) Performance comparison
├── setup.py         # pip install -e plugins/my_model/
└── README.md
```

## Step 1: Config

Your config tells vLLM how to initialize the model. Key trick: set `num_hidden_layers=0` if you want to skip KV cache allocation (encoder-only models don't need it).

```python
# config.py
from transformers import PretrainedConfig

class MyConfig(PretrainedConfig):
    model_type = "my_model"

    def __init__(self, hidden_size=768, num_hidden_layers=0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers  # 0 = no KV cache
```

## Step 2: Model

The model class needs two things: the `@attn_type("encoder_only")` decorator and `self.pooler` attribute.

```python
# model.py
from vllm.model_executor.models.interfaces_base import attn_type, default_pooling_type

@attn_type("encoder_only")
@default_pooling_type(tok_pooling_type="ALL")
class MyModel(nn.Module):
    is_pooling_model = True

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.encoder = YourEncoder(cfg)  # Your backbone
        self.pooler = YourPooler(cfg)    # Must be self.pooler

    def forward(self, input_ids, positions=None, **kwargs):
        return self.encoder(input_ids)

    def load_weights(self, weights):
        # Map checkpoint keys → model parameters
        ...
```

### vLLM discovers your pooler via `self.pooler`

vLLM looks for `self.pooler` on your model instance. If found, it calls `pooler.forward(hidden_states, pooling_metadata)` after your model's forward pass.

## Step 3: Pooler

Two options:

### Option A: Use a shared pooler from `poolers/`

```python
from poolers.colbert import ColBERTPooler
self.pooler = ColBERTPooler(hidden_size=768, output_dim=128)
```

Shared poolers (`poolers/colbert.py`, `poolers/gliner.py`, `poolers/colpali.py`) are reusable across multiple plugins.

### Option B: Plugin-specific pooler

When your pooler is tightly coupled to your model (e.g., it needs access to model components like a separate encoder), put it in `plugins/my_model/pooler.py`.

Example: the GLiNER-Linker pooler references the model's LSTM, scorer, and labels encoder — it can't be shared.

```python
# pooler.py
class MyPooler(nn.Module):
    def forward(self, hidden_states, pooling_metadata):
        # Extract per-sequence hidden states
        # Apply your scoring/pooling logic
        # Return list of tensors (one per sequence)
        return [result_tensor_1, result_tensor_2, ...]
```

## Step 4: Registration

```python
# __init__.py
from vllm import ModelRegistry
from transformers import AutoConfig
from .model import MyModel
from .config import MyConfig

AutoConfig.register("my_model", MyConfig, exist_ok=True)
ModelRegistry.register_model("MyModel", MyModel)
```

## Step 5: setup.py

```python
from setuptools import setup
setup(
    name="vllm-my-model",
    packages=["plugins.my_model"],
    entry_points={"vllm.general_plugins": ["my_model = plugins.my_model:register"]},
)
```

The `entry_points` line makes vLLM auto-discover your plugin on `pip install`.

## Step 6: Parity Test

Every plugin should have a parity test that compares vLLM output against a reference implementation:

```python
# parity_test.py
# Phase 1: Generate reference output using HuggingFace/original library
# Phase 2: Run vLLM pipeline and compare (cosine similarity, max diff)
```

See any existing plugin's `parity_test.py` for the pattern.

## Step 7: Processor (Optional)

For HTTP API users, add a processor that handles tokenization and output decoding:

```python
# processor.py
from forge.processor_base import BaseProcessor, PreprocessedInput

class MyProcessor(BaseProcessor):
    def preprocess(self, text, **kwargs):
        # Tokenize, build extra_kwargs
        return PreprocessedInput(prompt=..., pooling_params=..., metadata=...)

    def postprocess(self, raw_output, metadata=None):
        # Decode raw tensor into structured result
        return {"entities": [...]}
```

## Common Pitfalls

1. **Circular references**: If your pooler stores `self.model = model`, calling `model.eval()` causes infinite recursion. Use `object.__setattr__` to store component references without nn.Module registration.

2. **KV cache allocation**: Set `num_hidden_layers=0` in your config if your model is encoder-only. Otherwise vLLM allocates GPU memory for a KV cache you'll never use.

3. **Batch processing**: vLLM concatenates multiple sequences into a 1D tensor. Use the `positions` tensor to find sequence boundaries (it resets to 0 at each boundary). Pad and batch for GPU parallelism.

4. **Weight key mapping**: HuggingFace checkpoints use different key prefixes than your model. Map them in `load_weights()`.
