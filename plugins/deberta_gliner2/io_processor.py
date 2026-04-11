"""
IOProcessor plugin for deberta_gliner2 — server-side GLiNER2 extraction
via vLLM's native IOProcessor pipeline.

Uses the schema-based preprocessing from deberta_gliner2.processor instead of
the GLiNERPreprocessor/GLiNERDecoder used by other GLiNER plugins.

Supports four task types: entities, classification, relations, json.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  deberta_gliner2_io

Request format (online POST /pooling):
    {"data": {"text": "...", "labels": [...],
              "schema": {...},
              "threshold": 0.5,
              "include_confidence": false,
              "include_spans": false},
     "model": "...", "task": "plugin"}

Request format (offline):
    llm.encode({"data": {"text": "...", "schema": {...}}})
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict

from transformers import AutoTokenizer
from vllm.config import VllmConfig

from plugins.deberta_gliner2.processor import (
    build_special_token_ids,
    build_tokenization_cache,
    decode_output,
    format_results,
    normalize_gliner2_schema,
    preprocess,
)
from vllm_factory.io.base import FactoryIOProcessor, PoolingRequestOutput, PromptType, TokensPrompt


@dataclass
class GLiNER2Input:
    text: str
    schema: Dict = field(default_factory=dict)
    threshold: float = 0.5
    include_confidence: bool = False
    include_spans: bool = False
    truncate_overflow_text: bool = False


class DeBERTaGLiNER2IOProcessor(FactoryIOProcessor):
    """IOProcessor for deberta_gliner2 — schema-based extraction with DeBERTa backbone.

    Data flow:
        IOProcessorRequest(data={text, schema, threshold, include_confidence, include_spans})
        → factory_parse   → GLiNER2Input (with normalized schema)
        → factory_pre_process → TokensPrompt (+ stash extra_kwargs and metadata)
        → merge_pooling_params → PoolingParams(task="plugin", extra_kwargs=...)
        → engine.encode    → PoolingRequestOutput
        → factory_post_process → dict (decoded + formatted results)
    """

    pooling_task = "plugin"

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)

        model_id = vllm_config.model_config.model
        self._max_model_len = getattr(vllm_config.model_config, "max_model_len", None)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        self._tokenization_cache = build_tokenization_cache(self._tokenizer)
        self._special_token_ids = build_special_token_ids(self._tokenizer)
        self._schema_preprocess_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    @staticmethod
    def _coerce_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"'{field_name}' must be a boolean")

    # ------------------------------------------------------------------
    # FactoryIOProcessor implementation
    # ------------------------------------------------------------------

    def factory_parse(self, data: Any) -> GLiNER2Input:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError("Expected request data dict")

        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("'text' is required")

        threshold = data.get("threshold", 0.5)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("'threshold' must be a number") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("'threshold' must be between 0 and 1")

        include_confidence = self._coerce_bool(
            data.get("include_confidence", False), "include_confidence"
        )
        include_spans = self._coerce_bool(data.get("include_spans", False), "include_spans")
        truncate_overflow_text = self._coerce_bool(
            data.get("truncate_overflow_text", False), "truncate_overflow_text"
        )

        raw_schema = data.get("schema")
        labels = data.get("labels")

        if raw_schema is not None:
            schema = normalize_gliner2_schema(raw_schema)
        elif labels is not None:
            schema = normalize_gliner2_schema({"entities": labels})
        else:
            raise ValueError("Request must include schema or labels")

        return GLiNER2Input(
            text=text,
            schema=schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            truncate_overflow_text=truncate_overflow_text,
        )

    def factory_pre_process(
        self,
        parsed_input: GLiNER2Input,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        result = preprocess(
            self._tokenizer,
            parsed_input.text,
            parsed_input.schema,
            max_model_len=self._max_model_len,
            truncate_overflow_text=parsed_input.truncate_overflow_text,
            special_token_ids=self._special_token_ids or None,
            tokenization_cache=self._tokenization_cache,
            schema_cache=self._schema_preprocess_cache,
        )

        ids_list = result["input_ids"]

        gliner_data = {
            "mapped_indices": result["mapped_indices"],
            "schema_count": result["schema_count"],
            "special_token_ids": result["special_token_ids"],
            "token_pooling": result["token_pooling"],
            "schema_dict": result["schema_dict"],
            "task_types": result["task_types"],
            "schema_tokens_list": result["schema_tokens_list"],
            "text_tokens": result["text_tokens"],
            "original_text": result["original_text"],
            "start_mapping": result["start_mapping"],
            "end_mapping": result["end_mapping"],
            "threshold": parsed_input.threshold,
        }

        postprocess_meta = {
            "schema_dict": result["schema_dict"],
            "task_types": result["task_types"],
            "schema_tokens_list": result["schema_tokens_list"],
            "text_tokens": result["text_tokens"],
            "original_text": result["original_text"],
            "start_mapping": result["start_mapping"],
            "end_mapping": result["end_mapping"],
            "threshold": parsed_input.threshold,
            "include_confidence": parsed_input.include_confidence,
            "include_spans": parsed_input.include_spans,
        }

        self._stash(extra_kwargs=gliner_data, request_id=request_id, meta=postprocess_meta)

        return TokensPrompt(prompt_token_ids=ids_list)

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> Dict:
        if not model_output or request_meta is None:
            return {}

        output = model_output[0]
        raw = output.outputs.data
        if raw is None:
            return {}

        results = decode_output(
            raw,
            schema=request_meta["schema_dict"],
            task_types=request_meta["task_types"],
        )

        return format_results(
            results,
            threshold=request_meta.get("threshold", 0.5),
            include_confidence=request_meta.get("include_confidence", False),
            include_spans=request_meta.get("include_spans", False),
        )


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.deberta_gliner2.io_processor.DeBERTaGLiNER2IOProcessor"
