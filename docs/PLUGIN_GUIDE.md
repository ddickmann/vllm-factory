# Plugin Integration Guide

Central reference for building vLLM Factory plugins. Every plugin lives in
`plugins/<name>/` with the same 8-file structure.

---

## Decision Tree — Which Pattern Do I Need?

```
Does your model already exist in vLLM?
├── YES → Do you need a custom encoder with Triton kernels?
│   ├── YES → Use a custom model from models/ (see moderncolbert, mmbert_gliner)
│   └── NO  → Extend the built-in model directly (see colqwen3, collfm2)
└── NO  → Write a new model in models/ first (see models/modernbert/, models/mt5/)

Is your pooler reusable across multiple plugins?
├── YES → Put it in poolers/ and re-export from plugin (see poolers/gliner.py)
└── NO  → Put it in the plugin directory (task-specific, non-shared)

What kind of output do you need?
├── Multi-vector embeddings → pooler_for_token_embed + "ALL" pooling (see colqwen3)
├── Span extraction / NER   → Custom pooler with extra_kwargs (see mmbert_gliner)
├── Token classification    → pooler_for_token_classify (built-in vLLM pooler)
└── Relation / pairwise     → Custom pooler with bi-affine head (plugin-specific)
```

---

## Standard 8-File Structure

Every plugin has exactly these files:

| File | Purpose | Template |
|------|---------|----------|
| `__init__.py` | Auto-register model + config with vLLM on import | [below](#initpy) |
| `config.py` | Extend a HF config with task-specific params | [below](#configpy) |
| `model.py` | Wire backbone + pooler + weight loading | [below](#modelpy) |
| `pooler.py` | Re-export shared pooler OR define custom pooler | [below](#poolerpy) |
| `processor.py` | Async inference pipeline (preprocess → engine → postprocess) | [below](#processorpy) |
| `setup.py` | Make plugin pip-installable | [below](#setuppy) |
| `README.md` | Architecture, usage, benchmarks | — |
| `benchmark.py` | Async throughput benchmark | — |

---

## Templates

### `__init__.py`

```python
"""My Plugin — registers with vLLM on import."""
from vllm import ModelRegistry
from transformers import AutoConfig
from .model import MyModel
from .config import MyConfig

def register():
    try: AutoConfig.register("my_model_type", MyConfig, exist_ok=True)
    except Exception: pass
    try: ModelRegistry.register_model("MyModel", MyModel)
    except (KeyError, ValueError): pass

register()
__all__ = ["MyModel", "MyConfig"]
```

### `config.py`

```python
from transformers import SomeBaseConfig  # e.g. Qwen2Config, MT5Config, ModernBertConfig

class MyConfig(SomeBaseConfig):
    model_type = "my_model_type"  # must match __init__.py registration

    def __init__(self, my_custom_param: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.my_custom_param = my_custom_param
```

### `model.py`

```python
from vllm.config import VllmConfig

class MyModel(nn.Module):
    is_pooling_model = True  # required for pooling task

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # 1. Create backbone
        self.model = SomeBackbone(vllm_config=vllm_config, prefix=...)
        # 2. Create pooler
        self.pooler = some_pooler(...)

    def forward(self, input_ids, positions, **kwargs) -> torch.Tensor:
        return self.model(input_ids=input_ids, positions=positions, **kwargs)

    def load_weights(self, weights) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

### `pooler.py`

**Option A — Shared pooler (reusable):**
```python
from poolers.gliner import GLiNERSpanPooler
__all__ = ["GLiNERSpanPooler"]
```

**Option B — Custom pooler (plugin-specific):**
```python
class MyCustomPooler(nn.Module):
    def forward(self, hidden_states, pooling_metadata):
        # Custom logic here
        return outputs
```

### `processor.py`

Every plugin gets a processor that wraps the vLLM engine with a 3-stage pipeline:

```
User Input → preprocess() → AsyncLLMEngine.encode() → postprocess() → Output
```

Extend `forge.processor_base.BaseProcessor` and implement `preprocess()` + `postprocess()`:

```python
from forge.processor_base import BaseProcessor, PreprocessedInput
from vllm import PoolingParams
from vllm.inputs import TokensPrompt

class MyProcessor(BaseProcessor):
    """Async processor for my plugin."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess(self, text: str, **kwargs) -> PreprocessedInput:
        tokens = self._tokenizer(text, truncation=True, max_length=512)
        return PreprocessedInput(
            prompt=TokensPrompt(prompt_token_ids=tokens["input_ids"]),
            pooling_params=PoolingParams(task="token_embed"),
            metadata={"text": text},
        )

    def postprocess(self, raw_output, metadata=None):
        return torch.as_tensor(raw_output) if raw_output is not None else None
```

**What the base class gives you for free:**

| Method | Description |
|--------|-------------|
| `_ensure_engine()` | Lazy AsyncLLMEngine init with asyncio.Lock |
| `process_single(input, **kw)` | preprocess → encode → postprocess (with retry) |
| `process_batch(inputs, **kw)` | Concurrent `asyncio.gather` over `process_single` |
| `run(input)` / `run_batch(inputs)` | Sync wrappers via `asyncio.run()` |
| `close()` | Shutdown engine + free GPU memory |

**Usage:**

```python
processor = MyProcessor("my-model")

# Async
result = await processor.process_single("Hello world")
results = await processor.process_batch(["text1", "text2"])

# Sync
result = processor.run("Hello world")
results = processor.run_batch(["text1", "text2"])
```

### `setup.py`

```python
from setuptools import setup, find_packages
setup(
    name="vllm-factory-my-plugin", version="0.1.0",
    packages=find_packages(), python_requires=">=3.11",
    install_requires=["vllm==0.15.1", "torch>=2.0", "transformers>=4.40"],
    entry_points={"vllm.general_plugins": ["my_plugin = my_plugin:register"]},
)
```

---

## Weight Loading Patterns

### Pattern 1: AutoWeightsLoader + WeightsMapper (simplest)

Use when your model extends a vLLM built-in and only needs prefix remapping.

```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "custom_text_proj.": "projection.linear.",
    }
)

def load_weights(self, weights):
    loader = AutoWeightsLoader(self)
    return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

**Used by:** `colqwen3`, `collfm2`

### Pattern 2: Manual key mapping

Use when HF checkpoint key names differ structurally from vLLM's.

```python
def load_weights(self, weights):
    params = dict(self.model.named_parameters())
    for name, tensor in weights:
        target = self._map_key(name)  # your mapping logic
        if target and target in params:
            params[target].data.copy_(tensor)
```

**Used by:** `mmbert_gliner`, `mt5_gliner`

### Pattern 3: Separate head checkpoint

Use when pooler weights are in a separate file (e.g., `custom_head.pt`).

```python
head_path = Path(model_dir) / "custom_head.pt"
if head_path.exists():
    self.pooler.load_state_dict(torch.load(head_path), strict=False)
```

### Pattern 4: Weight remapping in iterator

Use when only a few keys need renaming.

```python
def remap():
    for name, tensor in weights:
        if name.startswith("score."):
            yield name.replace("score.", "classifier."), tensor
        else:
            yield name, tensor
loader.load_weights(remap())
```



---

## Reference Plugins by Complexity

| Plugin | Backbone | Pooler | Weight Loading | Complexity |
|--------|----------|--------|---------------|------------|
| `colqwen3` | vLLM built-in | Inline projection | WeightsMapper | ⭐⭐ |
| `collfm2` | vLLM built-in | Inline projection | WeightsMapper | ⭐⭐ |
| `moderncolbert` | Custom encoder | Shared (ColBERT) | Manual mapping | ⭐⭐⭐ |
| `mmbert_gliner` | Custom encoder | Shared (GLiNER) | Manual mapping | ⭐⭐⭐ |
| `mt5_gliner` | Custom encoder | Shared (GLiNER) | Manual + projection | ⭐⭐⭐⭐ |

---

## Registration Flow

```
pip install -e plugins/my_plugin/
    ↓
vLLM loads entry point: vllm.general_plugins → my_plugin:register
    ↓
register() calls:
  1. AutoConfig.register("my_model_type", MyConfig)
  2. ModelRegistry.register_model("MyModel", MyModel)
    ↓
vllm serve model-name --task pooling --trust-remote-code
    ↓
vLLM reads config.json → model_type → finds MyConfig
    ↓
vLLM instantiates MyModel(vllm_config=...) → loads weights → serves
```

---

## Parity Testing

Every plugin must prove its vLLM output matches the reference HuggingFace /
PyTorch implementation. Use `forge.testing.harness.ModelTestHarness`.

### Using ModelTestHarness

```python
from forge.testing.harness import ModelTestHarness

harness = ModelTestHarness(
    plugin_name="moderncolbert",
    model_id="answerdotai/ModernBERT-base",
)

# Define reference and vLLM functions
def reference_fn(inputs: list[str]) -> torch.Tensor:
    """Run the HuggingFace / PyTorch reference model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    tokens = tokenizer(inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        return model(**tokens).last_hidden_state

def vllm_fn(inputs: list[str]) -> torch.Tensor:
    """Run the vLLM model via LLM.encode()."""
    from vllm import LLM
    llm = LLM(model=model_id, task="pooling", trust_remote_code=True)
    outputs = llm.encode(inputs)
    return torch.stack([torch.tensor(o.outputs.embedding) for o in outputs])

# Run parity check
result = harness.test_parity(
    inputs=["Hello world", "The quick brown fox"],
    reference_fn=reference_fn,
    vllm_fn=vllm_fn,
    min_cosine_sim=0.99,  # Minimum cosine similarity to pass
    atol=1e-4,            # Absolute tolerance
)
assert result.passed, f"Parity failed: cosine_sim={result.cosine_similarity}"
```

### Parity Thresholds

| Precision | min_cosine_sim | atol | Notes |
|-----------|---------------|------|-------|
| FP32 | 0.9999 | 1e-4 | Near-exact match |
| FP16 / BF16 | 0.999 | 1e-3 | Standard for serving |
| FP8 | 0.99 | 1e-2 | Quantized models |

### What to Test

1. **Shape parity** — output dimensions match reference
2. **Value parity** — cosine similarity above threshold
3. **Edge cases** — empty input, max-length input, batch of 1 vs many
4. **Weight loading** — all expected weights loaded (check `load_weights` return set)

---

## Benchmarking

Every plugin includes a `benchmark.py` for async HTTP throughput measurement.

### benchmark.py Template

```python
"""My Plugin Throughput Benchmark"""
import argparse, asyncio, time, statistics
import aiohttp

async def send_request(session, url, model, text):
    """Send a single pooling request, return latency in ms."""
    start = time.perf_counter()
    async with session.post(url, json={"model": model, "input": text}) as r:
        await r.json()
    return (time.perf_counter() - start) * 1000

async def run_benchmark(base_url, model, num_requests, concurrency, warmup=50):
    url = f"{base_url}/v1/pooling"

    # Use a mix of short and long inputs for realistic benchmarking
    texts = [f"Short query {i}" for i in range(50)]
    texts += [f"Long document about topic {i}. " * 20 for i in range(50)]

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup — critical for accurate results (JIT, CUDA graphs, etc.)
        await asyncio.gather(*[
            send_request(session, url, model, texts[i % 100])
            for i in range(warmup)
        ])

        # Timed run with bounded concurrency
        start = time.perf_counter()
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded(text):
            async with semaphore:
                return await send_request(session, url, model, text)

        latencies = sorted(await asyncio.gather(*[
            bounded(texts[i % 100]) for i in range(num_requests)
        ]))
        elapsed = time.perf_counter() - start

    return {
        "req/s": round(num_requests / elapsed, 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--num-requests", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=32)
    args = p.parse_args()
    print(asyncio.run(run_benchmark(args.base_url, args.model,
                                     args.num_requests, args.concurrency)))

if __name__ == "__main__":
    main()
```

### Running Benchmarks

```bash
# 1. Start the vLLM server
vllm serve my-model --task pooling --trust-remote-code &

# 2. Wait for server to be ready
sleep 30

# 3. Run the benchmark
python plugins/my_plugin/benchmark.py \
    --model my-model \
    --num-requests 1000 \
    --concurrency 32
```

### Using ModelTestHarness for Offline Benchmarks

For GPU-level benchmarking without HTTP overhead:

```python
harness = ModelTestHarness("my_plugin", "my-model-id")

results = harness.benchmark_throughput(
    inputs=["Sample text"] * 100,
    run_fn=lambda batch: llm.encode(batch),
    batch_sizes=[1, 8, 32, 128],
    n_warmup=3,
    n_runs=10,
)

# Generate markdown report
harness.generate_report("reports/my_plugin_report.md")
```

### Generating a Full Report

`ModelTestHarness.generate_report()` outputs a markdown file with both parity
and benchmark tables:

```python
harness.test_parity(inputs, reference_fn, vllm_fn)
harness.benchmark_throughput(inputs, run_fn)
harness.generate_report("reports/my_plugin.md")

# Output includes:
# ## Parity Results
# | # | Cosine Sim | Max Error | Mean Error | Status |
#
# ## Benchmark Results
# | Batch Size | Tokens/sec | P50 (ms) | P95 (ms) | P99 (ms) |
```

### Benchmark Checklist

- [ ] Warmup before timing (50+ requests for HTTP, 3+ runs for offline)
- [ ] Test multiple concurrency levels (1, 8, 32)
- [ ] Mix short and long inputs
- [ ] Report p50 AND p99 (mean is misleading)
- [ ] Compare against reference (PyTorch / PyLate / sentence-transformers)
- [ ] Record GPU memory usage (via `nvidia-smi`)

---

## Common Gotchas

| Issue | Solution |
|-------|----------|
| `KeyError: model not registered` | Ensure `register()` is called in `__init__.py` AND `setup.py` entry point |
| `attention_bias mismatch` (Qwen3) | Use `Qwen3Model`, not `Qwen2Model` — they differ on `attention_bias` |
| `Weight shape mismatch` | Check `WeightsMapper` prefix order (longer prefixes first) |
| `NaN in classification` | Return hidden states from `forward()`, NOT logits — the pooler applies the head |
| `Custom head not loading` | Check that `custom_head.pt` exists in the model directory |
| `Kernel import error` | Use absolute imports: `from kernels.ff_fused import ...` |
