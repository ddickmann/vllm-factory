"""
IOProcessor plugin for deberta_gliner2 — server-side GLiNER2 extraction
via vLLM's native IOProcessor pipeline.

Uses the schema-based preprocessing from deberta_gliner2.processor instead of
the GLiNERPreprocessor/GLiNERDecoder used by other GLiNER plugins.

Supports four task types: entities, classification, relations, json.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  deberta_gliner2_io

Request format (online POST /pooling):
    {"data": {"text": "...", "labels": ["person", "org"],
              "task_type": "entities"},
     "model": "...", "task": "plugin"}

Request format (offline):
    llm.encode({"data": {"text": "...", "labels": [...]}})
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict

from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm_factory.io.base import FactoryIOProcessor, TokensPrompt, PromptType, PoolingRequestOutput

from plugins.deberta_gliner2.processor import (
    build_schema_for_classification,
    build_schema_for_entities,
    build_schema_for_json,
    build_schema_for_relations,
    decode_output,
    format_results,
    preprocess,
)

_SCHEMA_BUILDERS = {
    "entities": build_schema_for_entities,
    "classification": build_schema_for_classification,
    "relations": build_schema_for_relations,
    "json": build_schema_for_json,
}


@dataclass
class GLiNER2Input:
    """Validated extraction request after parse_request."""

    text: str
    labels: Any
    task_type: str
    schema: Dict = field(default_factory=dict)


class DeBERTaGLiNER2IOProcessor(FactoryIOProcessor):
    """IOProcessor for deberta_gliner2 — schema-based extraction with DeBERTa backbone.

    Data flow:
        IOProcessorRequest(data={text, labels, task_type})
        → factory_parse   → GLiNER2Input (with built schema)
        → factory_pre_process → TokensPrompt (+ stash extra_kwargs and metadata)
        → merge_pooling_params → PoolingParams(task="plugin", extra_kwargs=...)
        → engine.encode    → PoolingRequestOutput
        → factory_post_process → dict (decoded + formatted results)
    """

    pooling_task = "plugin"

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)

        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )

    # ------------------------------------------------------------------
    # Task-type auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_task_type(data: dict) -> str:
        if "task_type" in data:
            return data["task_type"]
        labels = data.get("labels", [])
        if isinstance(labels, dict):
            first_val = next(iter(labels.values()), None) if labels else None
            if isinstance(first_val, dict) and ("head" in first_val or "tail" in first_val):
                return "relations"
            if isinstance(first_val, (list, dict)):
                return "classification"
            return "json"
        return "entities"

    # ------------------------------------------------------------------
    # FactoryIOProcessor implementation
    # ------------------------------------------------------------------

    def factory_parse(self, data: Any) -> GLiNER2Input:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' and 'labels' keys, got {type(data)}")

        labels = data.get("labels", [])
        if not labels:
            raise ValueError("'labels' must not be empty")

        task_type = self._detect_task_type(data)
        builder = _SCHEMA_BUILDERS.get(task_type)
        if builder is None:
            raise ValueError(
                f"Unknown task_type '{task_type}'. Must be one of: {list(_SCHEMA_BUILDERS)}"
            )

        schema = builder(labels)

        return GLiNER2Input(
            text=data.get("text", ""),
            labels=labels,
            task_type=task_type,
            schema=schema,
        )

    def factory_pre_process(
        self,
        parsed_input: GLiNER2Input,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        result = preprocess(self._tokenizer, parsed_input.text, parsed_input.schema)

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
        }

        postprocess_meta = {
            "schema_dict": result["schema_dict"],
            "task_types": result["task_types"],
            "schema_tokens_list": result["schema_tokens_list"],
            "text_tokens": result["text_tokens"],
            "original_text": result["original_text"],
            "start_mapping": result["start_mapping"],
            "end_mapping": result["end_mapping"],
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

        return format_results(results)


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.deberta_gliner2.io_processor.DeBERTaGLiNER2IOProcessor"
