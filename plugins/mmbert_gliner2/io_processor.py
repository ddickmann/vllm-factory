"""
IOProcessor plugin for mmbert_gliner2 — ModernBERT + GLiNER2 schema-based extraction.

Reuses the deberta_gliner2 processor logic (schema normalization, tokenization,
postprocessing) with the same request format.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  mmbert_gliner2_io

Request format (online POST /pooling):
    {"data": {"text": "...", "schema": {...}, "threshold": 0.4},
     "model": "...", "task": "plugin"}
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
    raw_schema: Dict = field(default_factory=dict)


class ModernBertGLiNER2IOProcessor(FactoryIOProcessor):
    """IOProcessor for mmbert_gliner2 — GLiNER2 extraction with ModernBERT backbone.

    Data flow:
        IOProcessorRequest(data={text, schema, threshold, ...})
        → factory_parse → GLiNER2Input (with normalized schema)
        → factory_pre_process → TokensPrompt (+ stash extra_kwargs and metadata)
        → merge_pooling_params → PoolingParams(task="plugin", extra_kwargs=...)
        → engine.encode → PoolingRequestOutput
        → factory_post_process → structured extraction results
    """

    pooling_task = "plugin"

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        from plugins.deberta_gliner_linker.vllm_pooling_attention_mask import (
            apply_pooling_attention_mask_patch,
        )

        apply_pooling_attention_mask_patch()

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

        raw_schema = data.get("schema")
        labels = data.get("labels")

        if raw_schema is not None:
            schema = normalize_gliner2_schema(raw_schema)
        elif labels is not None:
            raw_schema = {"entities": labels}
            schema = normalize_gliner2_schema(raw_schema)
        else:
            raise ValueError("Request must include schema or labels")

        return GLiNER2Input(
            text=text,
            schema=schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            raw_schema=raw_schema,
        )

    def factory_pre_process(
        self,
        parsed_input: GLiNER2Input,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        result = preprocess(
            tokenizer=self._tokenizer,
            text=parsed_input.text,
            schema=parsed_input.schema,
            token_pooling=getattr(self, "_token_pooling", "first"),
            max_model_len=self._max_model_len,
            special_token_ids=self._special_token_ids or None,
            tokenization_cache=self._tokenization_cache,
            schema_cache=self._schema_preprocess_cache,
        )

        input_ids = result["input_ids"]
        gliner_data = {
            "input_ids": input_ids,
            "mapped_indices": result["mapped_indices"],
            "special_token_ids": result["special_token_ids"],
            "token_pooling": result["token_pooling"],
            "schema_dict": result["schema_dict"],
            "schema_tokens_list": result["schema_tokens_list"],
            "task_types": result["task_types"],
            "text_tokens": result["text_tokens"],
            "schema_count": result["schema_count"],
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
            "schema_count": result["schema_count"],
            "original_text": result["original_text"],
            "start_mapping": result["start_mapping"],
            "end_mapping": result["end_mapping"],
            "threshold": parsed_input.threshold,
            "include_confidence": parsed_input.include_confidence,
            "include_spans": parsed_input.include_spans,
            "raw_schema": getattr(parsed_input, "raw_schema", parsed_input.schema),
        }

        self._stash(extra_kwargs=gliner_data, request_id=request_id, meta=postprocess_meta)

        return TokensPrompt(prompt_token_ids=input_ids)

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> Any:
        if not model_output or request_meta is None:
            return []

        output = model_output[0]
        raw = output.outputs.data
        if raw is None:
            return []

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
    return "plugins.mmbert_gliner2.io_processor.ModernBertGLiNER2IOProcessor"
