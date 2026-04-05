"""Vanilla baseline runners — reference implementations for each model family.

Each runner takes a list of inputs and returns (outputs, latencies_ms) where
outputs are the raw predictions and latencies_ms is a list of per-request
latency measurements.
"""

from __future__ import annotations

import os
import time
from typing import Any


def _timed_batch(fn, inputs: list, n_warmup: int = 3, n_runs: int = 10) -> tuple[Any, list[float]]:
    """Run fn on inputs with warmup, return (last_output, latencies_ms)."""
    for _ in range(n_warmup):
        fn(inputs)

    latencies = []
    output = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        output = fn(inputs)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
    return output, latencies


# ---------------------------------------------------------------------------
# SentenceTransformers (EmbeddingGemma, generic dense embeddings)
# ---------------------------------------------------------------------------

class SentenceTransformersRunner:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_id, trust_remote_code=True)
        return self._model

    def encode_batch(self, texts: list[str]):
        model = self._load()
        return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    def run(self, texts: list[str], n_warmup: int = 3, n_runs: int = 10):
        return _timed_batch(self.encode_batch, texts, n_warmup, n_runs)

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
            _try_cuda_empty()


# ---------------------------------------------------------------------------
# HF AutoModel (LFM2-ColBERT — raw encoder + projection)
# ---------------------------------------------------------------------------

class HFAutoModelRunner:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._proj_weight = None

    def _load(self):
        if self._model is not None:
            return
        import safetensors.torch
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self._model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).cuda().eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        proj_path = hf_hub_download(repo_id=self.model_id, filename="1_Dense/model.safetensors")
        with safetensors.torch.safe_open(proj_path, framework="pt", device="cpu") as f:
            self._proj_weight = f.get_tensor("linear.weight").to(torch.bfloat16).cuda()

    def encode_batch(self, texts: list[str]):
        import torch
        self._load()
        embeddings = []
        for text in texts:
            tokens = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=False
            )
            tokens = {k: v.cuda() for k, v in tokens.items()}
            with torch.no_grad():
                out = self._model(**tokens)
            hidden = out.last_hidden_state.squeeze(0)
            projected = hidden @ self._proj_weight.T
            projected = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            embeddings.append(projected.cpu().float())
        return embeddings

    def run(self, texts: list[str], n_warmup: int = 3, n_runs: int = 10):
        return _timed_batch(self.encode_batch, texts, n_warmup, n_runs)

    def cleanup(self):
        del self._model, self._tokenizer, self._proj_weight
        self._model = self._tokenizer = self._proj_weight = None
        _try_cuda_empty()


# ---------------------------------------------------------------------------
# GLiNER (mt5_gliner, deberta_gliner, mmbert_gliner)
# ---------------------------------------------------------------------------

class GLiNERRunner:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from gliner import GLiNER
            self._model = GLiNER.from_pretrained(
                self.model_id,
                map_location="cpu",
            )
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()
        return self._model

    def predict_batch(self, inputs: list[dict]):
        model = self._load()
        texts = [inp["text"] for inp in inputs]
        all_labels = sorted({label for inp in inputs for label in inp["labels"]})
        per_sample_labels = [set(inp["labels"]) for inp in inputs]
        batch_results = model.inference(
            texts, all_labels,
            batch_size=max(1, len(inputs)),
            threshold=0.5,
            flat_ner=True,
        )
        return [
            [e for e in entities if e.get("label", "") in sample_labels]
            for entities, sample_labels in zip(batch_results, per_sample_labels)
        ]

    def run(self, inputs: list[dict], n_warmup: int = 3, n_runs: int = 10):
        return _timed_batch(self.predict_batch, inputs, n_warmup, n_runs)

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
            _try_cuda_empty()


# ---------------------------------------------------------------------------
# GLiNER2 (deberta_gliner2 — fastino/gliner2-large-v1)
# ---------------------------------------------------------------------------

class GLiNER2Runner:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from gliner2 import GLiNER2
            os.environ.setdefault("USE_FLASHDEBERTA", "1")
            self._model = GLiNER2.from_pretrained(self.model_id)
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()
        return self._model

    def predict_batch(self, inputs: list[dict]):
        model = self._load()
        texts = [inp["text"] for inp in inputs]
        schemas = [model.create_schema().entities(inp["labels"]) for inp in inputs]
        return model.batch_extract(
            texts,
            schemas,
            batch_size=max(1, len(inputs)),
            threshold=0.5,
            num_workers=0,
            format_results=True,
            include_confidence=False,
            include_spans=False,
        )

    def run(self, inputs: list[dict], n_warmup: int = 3, n_runs: int = 10):
        return _timed_batch(self.predict_batch, inputs, n_warmup, n_runs)

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
            _try_cuda_empty()


# ---------------------------------------------------------------------------
# GLinker (deberta_gliner_linker L3, modernbert_gliner_rerank L4)
# ---------------------------------------------------------------------------

class GLinkerRunner:
    """Vanilla runner using GLinker L3/L4 components for NER."""

    def __init__(self, model_id: str, layer: str = "l3"):
        self.model_id = model_id
        self.layer = layer
        self._component = None

    def _load(self):
        if self._component is not None:
            return self._component
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.layer == "l4":
            from glinker.l4 import L4Component, L4Config
            config = L4Config(
                model_name=self.model_id, device=device,
                threshold=0.5, flat_ner=True,
            )
            self._component = L4Component(config)
        else:
            from glinker.l3 import L3Component, L3Config
            config = L3Config(
                model_name=self.model_id, device=device,
                threshold=0.5, flat_ner=True,
            )
            self._component = L3Component(config)
        return self._component

    def predict_batch(self, inputs: list[dict]):
        comp = self._load()
        texts = [inp["text"] for inp in inputs]
        all_labels = sorted({label for inp in inputs for label in inp["labels"]})
        batch_results = comp.model.inference(
            texts, all_labels,
            batch_size=max(1, len(inputs)),
            threshold=0.5,
            flat_ner=True,
        )
        return batch_results

    def run(self, inputs: list[dict], n_warmup: int = 3, n_runs: int = 10):
        return _timed_batch(self.predict_batch, inputs, n_warmup, n_runs)

    def cleanup(self):
        if self._component is not None:
            del self._component
            self._component = None
            _try_cuda_empty()


# ---------------------------------------------------------------------------
# PyLate ColBERT (moderncolbert, colbert_zero, lfm2_colbert)
# ---------------------------------------------------------------------------

class PyLateColBERTRunner:
    def __init__(self, model_id: str, prompt_name: str | None = None):
        self.model_id = model_id
        self.prompt_name = prompt_name
        self._model = None

    def _load(self):
        if self._model is None:
            from pylate import models
            self._model = models.ColBERT(model_name_or_path=self.model_id)
        return self._model

    def encode_batch(self, inputs: list[dict]):
        model = self._load()
        texts = [inp["text"] for inp in inputs]
        is_query = inputs[0].get("is_query", False) if inputs else False
        kwargs: dict[str, Any] = {"is_query": is_query, "batch_size": len(texts)}
        if self.prompt_name and not is_query:
            kwargs["prompt_name"] = self.prompt_name
        return model.encode(texts, **kwargs)

    def run(self, inputs: list[dict], n_warmup: int = 3, n_runs: int = 10):
        return _timed_batch(self.encode_batch, inputs, n_warmup, n_runs)

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
            _try_cuda_empty()


# ---------------------------------------------------------------------------
# SauerkrautLM ColPali (collfm2, colqwen3)
# ---------------------------------------------------------------------------

class SauerkrautColPaliRunner:
    def __init__(self, model_id: str, model_class: str = "ColLFM2"):
        self.model_id = model_id
        self.model_class = model_class
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch

        if self.model_class == "ColQwen3":
            from sauerkrautlm_colpali.models import ColQwen3 as ModelCls
            from sauerkrautlm_colpali.models import ColQwen3Processor as ProcCls
        else:
            from sauerkrautlm_colpali.models import ColLFM2 as ModelCls
            from sauerkrautlm_colpali.models import ColLFM2Processor as ProcCls

        from transformers import AutoModelForImageTextToText
        _orig = AutoModelForImageTextToText.from_pretrained

        def _patched(*a, **kw):
            kw["attn_implementation"] = "sdpa"
            return _orig(*a, **kw)

        AutoModelForImageTextToText.from_pretrained = _patched
        try:
            self._model = (
                ModelCls.from_pretrained(self.model_id)
                .to(torch.bfloat16)
                .to("cuda")
                .eval()
            )
        finally:
            AutoModelForImageTextToText.from_pretrained = _orig

        self._processor = ProcCls.from_pretrained(self.model_id)

    def encode_batch(self, inputs: list[dict]):
        import torch
        from PIL import Image

        self._load()
        images = [Image.open(inp["image"]).convert("RGB") for inp in inputs]
        batch = self._processor.process_images(images)
        batch = {k: v.to(self._model.device) for k, v in batch.items()}
        batch.pop("token_type_ids", None)
        with torch.no_grad():
            return self._model(**batch)

    def run(self, inputs: list[dict], n_warmup: int = 3, n_runs: int = 10):
        return _timed_batch(self.encode_batch, inputs, n_warmup, n_runs)

    def cleanup(self):
        if self._model is not None:
            del self._model, self._processor
            self._model = self._processor = None
            _try_cuda_empty()


# ---------------------------------------------------------------------------
# Nemotron ColEmbed (transformers AutoModel)
# ---------------------------------------------------------------------------

class NemotronTransformersRunner:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(
                self.model_id,
                device_map="cuda",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
        return self._model

    def encode_batch(self, inputs: list[dict]):
        from PIL import Image
        model = self._load()
        images = [Image.open(inp["image"]).convert("RGB") for inp in inputs]
        return model.forward_images(images, batch_size=len(images))

    def run(self, inputs: list[dict], n_warmup: int = 3, n_runs: int = 10):
        return _timed_batch(self.encode_batch, inputs, n_warmup, n_runs)

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
            _try_cuda_empty()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_RUNNERS = {
    "sentence_transformers": SentenceTransformersRunner,
    "hf_automodel": HFAutoModelRunner,
    "gliner": GLiNERRunner,
    "gliner2": GLiNER2Runner,
    "glinker": GLinkerRunner,
    "pylate_colbert": PyLateColBERTRunner,
    "sauerkraut_colpali": SauerkrautColPaliRunner,
    "nemotron_transformers": NemotronTransformersRunner,
}


def get_runner(family: str, model_id: str, **kwargs):
    cls = _RUNNERS.get(family)
    if cls is None:
        raise KeyError(f"Unknown vanilla family {family!r}. Available: {list(_RUNNERS)}")
    return cls(model_id, **kwargs)


def _try_cuda_empty():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
