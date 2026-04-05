# Bring Your Own Pooler (BYOP)

Compose **any backbone** shipped in `models/` with **any pooler** — built-in or your own — without writing a model class.

---

## Tutorial: Build and deploy a custom pooler in 3 steps

### Step 1 — Write your pooler

A pooler is a plain Python class with two methods: `get_tasks()` and `forward()`.
It receives a flat tensor of hidden states from the backbone and a `PoolerContext`
with per-sequence metadata. It returns one embedding tensor per input sequence.

```python
# my_pooler.py
import torch
from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states


class WeightedAttentionPooler(torch.nn.Module):
    """Attention-weighted pooling — learns a query vector that attends
    over each sequence's token embeddings to produce a single vector."""

    def __init__(self, hidden_size: int = 768, temperature: float = 1.0, **kwargs):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(hidden_size))
        self.temperature = temperature

    def get_tasks(self) -> set[str]:
        return {"embed"}

    def forward(
        self,
        hidden_states: torch.Tensor,  # (total_tokens, hidden_dim) — all seqs concatenated
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        parts = split_hidden_states(hidden_states, ctx.seq_lengths)
        results = []
        for seq in parts:
            # seq shape: (seq_len, hidden_dim)
            logits = (seq @ self.query) / self.temperature
            weights = torch.softmax(logits, dim=0)
            pooled = (weights.unsqueeze(-1) * seq).sum(dim=0)
            results.append(pooled)
        return results
```

**Key rules:**
- `__init__` must accept `hidden_size: int` as a keyword argument (the backbone's hidden dimension).
- `forward` receives the *concatenated* hidden states for all sequences in the batch. Use `split_hidden_states()` to split them into per-sequence tensors.
- Return exactly one tensor per sequence (shape `(hidden_dim,)` for single-vector, or `(seq_len, dim)` for multi-vector poolers like ColBERT).

### Step 2 — Register your pooler

Before creating the vLLM engine, register your pooler under a name:

```python
from vllm_factory.composable.pooler_registry import register_pooler
from my_pooler import WeightedAttentionPooler

register_pooler("weighted_attention", WeightedAttentionPooler)
```

### Step 3 — Serve

Tell vLLM to use the composable model and your pooler:

```bash
VLLM_FACTORY_POOLER=weighted_attention \
vllm serve answerdotai/ModernBERT-base \
  --runner pooling \
  --trust-remote-code \
  --dtype bfloat16 \
  --model-impl ComposablePoolingModel
```

Or use it offline in Python:

```python
import my_pooler  # triggers register_pooler() call
import os

os.environ["VLLM_FACTORY_POOLER"] = "weighted_attention"

from vllm import LLM
llm = LLM(
    model="answerdotai/ModernBERT-base",
    task="embed",
    trust_remote_code=True,
    dtype="bfloat16",
)
out = llm.encode(["Hello world", "Another sentence"])
for i, r in enumerate(out):
    print(f"Sequence {i}: shape={r.outputs.data.shape}")
```

That's it. No model file. No plugin boilerplate. Just your pooler logic.

---

## Using a built-in pooler

If you don't need a custom pooler, pick one of the built-ins — no registration needed:

```bash
# Mean pooling over all tokens
VLLM_FACTORY_POOLER=mean vllm serve google/t5gemma-2-270m-270m \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --model-impl ComposablePoolingModel

# CLS token pooling (first token)
VLLM_FACTORY_POOLER=cls vllm serve answerdotai/ModernBERT-base \
  --runner pooling --trust-remote-code --dtype bfloat16 \
  --model-impl ComposablePoolingModel
```

### Built-in poolers

| Name | Description |
|------|-------------|
| `mean` | Average all token embeddings |
| `cls` | First token ([CLS]) embedding |
| `normalized_mean` | Mean pooling + L2 normalization |
| `colbert` | Token-level multi-vector (ColBERT-style) |
| `colpali` | Vision-language multi-vector (ColPali-style) |
| `passthrough` | Return all token embeddings unchanged |

---

## Supported backbones

Backbone resolution is automatic — `ComposablePoolingModel` reads `model_type` from the HuggingFace checkpoint's `config.json`.

| Backbone | Model Class | HF `model_type` |
|----------|------------|------------------|
| ModernBERT | `ModernBertModel` | `modernbert` |
| DeBERTa | `DebertaEncoderModel` | `deberta` |
| DeBERTa v2/v3 | `DebertaV2EncoderModel` | `deberta-v2` |
| mT5 | `MT5Encoder` | `mt5` |
| T5Gemma2 | `T5Gemma2Encoder` | `t5gemma2` |

---

## How it works

```
VLLM_FACTORY_POOLER=mean vllm serve <model> --model-impl ComposablePoolingModel
                          │
                          ▼
              ┌─────────────────────────┐
              │  ComposablePoolingModel  │
              │                         │
              │  backbone (auto)  ◀──── resolved from model_type in config.json
              │  pooler (env/cfg) ◀──── resolved from VLLM_FACTORY_POOLER
              │  VllmPoolerAdapter      │
              └─────────────────────────┘
```

1. **Backbone resolution**: `model_type` in `config.json` maps to a backbone in the registry. Each backbone has a `create_instance` callable that handles its specific constructor signature.
2. **Pooler resolution**: `VLLM_FACTORY_POOLER` env var has highest priority, then `pooler_type` in the model's `config.json`.
3. **Pooler instantiation**: Tries `(hidden_size=, **config)` first, falls back to `(hidden_size)`, then no-args.
4. **Weight loading**: Backbone weights go to the backbone. Weights prefixed with `pooler_head.` go to the pooler (if it's an `nn.Module` with learned parameters).

---

## The FactoryPooler protocol

Any pooler — built-in or custom — must implement this interface:

```python
import torch
from vllm_factory.pooling.protocol import PoolerContext

class MyPooler:
    def __init__(self, hidden_size: int = 768, **kwargs):
        ...

    def get_tasks(self) -> set[str]:
        """Return supported task names (typically {'embed'})."""
        ...

    def forward(
        self,
        hidden_states: torch.Tensor,     # (total_tokens, hidden_dim)
        ctx: PoolerContext,               # batch metadata
    ) -> list[torch.Tensor | None]:
        """Return one tensor per sequence in the batch."""
        ...
```

**`PoolerContext` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `seq_lengths` | `list[int]` | Token count per sequence |
| `extra_kwargs` | `list[dict]` | Per-sequence kwargs from the request |
| `tasks` | `list[str]` | Task name per sequence |
| `prompt_token_ids` | `list[Tensor]` | Original token IDs |

Use `split_hidden_states(hidden_states, ctx.seq_lengths)` to split the concatenated batch into per-sequence tensors.

---

## Stateless vs. learned poolers

**Stateless** (plain class — no `nn.Module`): mean, cls, normalized_mean. No GPU parameters, no weight loading needed.

**Learned** (`nn.Module` subclass): Has trainable parameters (e.g. a projection layer or attention query). Parameters are automatically tracked by PyTorch, moved to the correct device/dtype, and loaded from checkpoint weights prefixed with `pooler_head.`.

---

## For deployment: entry-point registration

Instead of calling `register_pooler()` manually, you can use Python entry points so vLLM discovers your pooler at startup:

```toml
# In your package's pyproject.toml
[project.entry-points."vllm.general_plugins"]
my_pooler = "my_package:register"
```

```python
# my_package/__init__.py
def register():
    from vllm_factory.composable.pooler_registry import register_pooler
    from my_package.pooler import WeightedAttentionPooler
    register_pooler("weighted_attention", WeightedAttentionPooler)
```

---

## Existing plugins are not affected

The composable path is a **parallel, opt-in** feature. All existing hardcoded plugins (GLiNER, ColBERT, ColPali, etc.) continue to work exactly as before. Use the composable path when you want to:

- Try different poolers on the same backbone without writing boilerplate
- Prototype a custom pooler quickly
- Deploy a backbone + pooler combination without a dedicated model file
