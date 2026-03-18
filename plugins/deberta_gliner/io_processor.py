"""
IOProcessor plugin for deberta_gliner — server-side GLiNER NER via vLLM's
native IOProcessor pipeline.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  deberta_gliner_io

Request format (online POST /pooling):
    {"data": {"text": "...", "labels": ["person", "org"], "threshold": 0.5},
     "model": "...", "task": "plugin"}

Request format (offline):
    llm.encode({"data": {"text": "...", "labels": [...]}})
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer

from vllm.config import VllmConfig
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
from vllm.inputs.data import PromptType
from vllm.inputs import TokensPrompt
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams

from forge.gliner_postprocessor import GLiNERDecoder, get_final_entities
from forge.gliner_preprocessor import GLiNERPreprocessor


@dataclass
class GLiNERInput:
    """Validated NER request after parse_request."""
    text: str
    labels: list[str]
    threshold: float = 0.5
    flat_ner: bool = False
    multi_label: bool = False


class DeBERTaGLiNERIOProcessor(IOProcessor[GLiNERInput, list[dict[str, Any]]]):
    """IOProcessor for deberta_gliner — GLiNER NER with DeBERTa backbone.

    Data flow:
        IOProcessorRequest(data={text, labels, ...})
        → parse_request → GLiNERInput
        → pre_process   → TokensPrompt (+ stash extra_kwargs and metadata)
        → validate_or_generate_params → PoolingParams(extra_kwargs=gliner_data)
        → engine.encode  → PoolingRequestOutput
        → post_process   → list[dict] (decoded entities)
        → output_to_response → IOProcessorResponse(data=[...])
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        from plugins.deberta_gliner_linker.vllm_pooling_attention_mask import (
            apply_pooling_attention_mask_patch,
        )
        apply_pooling_attention_mask_patch()

        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True,
        )
        config = vllm_config.model_config.hf_config

        self._preprocessor = GLiNERPreprocessor(
            underlying_tokenizer=self._tokenizer,
            config=config,
            device="cpu",
            include_attention_mask=True,
        )
        self._decoder = GLiNERDecoder()

        self._lock = threading.Lock()
        self._pending_extra_kwargs: dict | None = None
        self._request_meta: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # IOProcessor ABC implementation
    # ------------------------------------------------------------------

    def parse_request(self, request: Any) -> GLiNERInput:
        if hasattr(request, "data"):
            data = request.data
        elif isinstance(request, dict) and "data" in request:
            data = request["data"]
        else:
            data = request

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dict with 'text' and 'labels' keys, got {type(data)}"
            )

        labels = data.get("labels", [])
        if not labels:
            raise ValueError("'labels' list must not be empty")

        return GLiNERInput(
            text=data.get("text", ""),
            labels=labels,
            threshold=float(data.get("threshold", 0.5)),
            flat_ner=bool(data.get("flat_ner", False)),
            multi_label=bool(data.get("multi_label", False)),
        )

    def pre_process(
        self,
        prompt: GLiNERInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        result = self._preprocessor(prompt.text, prompt.labels, device="cpu")
        enc = result["model_inputs"]
        meta = result["postprocessing_metadata"]

        input_ids = enc["input_ids"][0]
        words_mask = enc["words_mask"][0]
        text_lengths = enc["text_lengths"][0].item()

        ids_list = input_ids.tolist()
        mask_list = words_mask.tolist()
        attn_list = enc["attention_mask"][0].tolist()

        gliner_data = {
            "input_ids": ids_list,
            "words_mask": mask_list,
            "text_lengths": text_lengths,
            "attention_mask": attn_list,
            "span_idx": enc["span_idx"][0].tolist(),
            "span_mask": enc["span_mask"][0].tolist(),
        }

        postprocess_meta = {
            "text": prompt.text,
            "labels": prompt.labels,
            "threshold": prompt.threshold,
            "flat_ner": prompt.flat_ner,
            "multi_label": prompt.multi_label,
            "tokens": meta["tokens"],
            "word_positions": meta["word_positions"],
            "id_to_classes": meta["id_to_classes"],
        }

        with self._lock:
            self._pending_extra_kwargs = gliner_data
            if request_id is not None:
                self._request_meta[request_id] = postprocess_meta
            else:
                self._request_meta["_offline"] = postprocess_meta

        return TokensPrompt(prompt_token_ids=ids_list)

    def validate_or_generate_params(
        self, params: PoolingParams | None = None,
    ) -> PoolingParams:
        with self._lock:
            extra = self._pending_extra_kwargs
            self._pending_extra_kwargs = None

        if params is not None and extra is not None:
            params.extra_kwargs = extra
            return params

        return PoolingParams(extra_kwargs=extra)

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        meta_key = request_id if request_id is not None else "_offline"
        with self._lock:
            meta = self._request_meta.pop(meta_key, None)

        if not model_output or meta is None:
            return []

        output = model_output[0]
        raw = output.outputs.data
        if raw is None:
            return []

        scores = torch.as_tensor(raw) if not isinstance(raw, torch.Tensor) else raw

        if scores.dim() == 1 and scores.numel() > 3:
            L = int(scores[0].item())
            K = int(scores[1].item())
            C = int(scores[2].item())
            logits = scores[3:].reshape(1, L, K, C)
        elif scores.dim() == 3:
            logits = scores.unsqueeze(0)
        else:
            return []

        decoded = self._decoder.decode(
            tokens=meta["tokens"],
            id_to_classes=meta["id_to_classes"],
            logits=logits,
            flat_ner=meta.get("flat_ner", False),
            threshold=meta.get("threshold", 0.5),
            multi_label=meta.get("multi_label", False),
        )

        entities_batch = get_final_entities(
            decoded_outputs=decoded,
            word_positions=meta["word_positions"],
            original_texts=[meta["text"]],
        )

        return entities_batch[0] if entities_batch else []

    def output_to_response(
        self, plugin_output: list[dict[str, Any]],
    ) -> IOProcessorResponse:
        return IOProcessorResponse(data=plugin_output)


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.deberta_gliner.io_processor.DeBERTaGLiNERIOProcessor"
