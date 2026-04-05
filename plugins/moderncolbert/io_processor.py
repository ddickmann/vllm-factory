"""
IOProcessor plugin for moderncolbert — ColBERT multi-vector embeddings via
vLLM's native IOProcessor pipeline.

Handles text queries and document inputs with [Q]/[D] prefix insertion,
returning multi-vector embeddings as a list of floats.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  moderncolbert_io

Request format (online POST /pooling):
    Query: {"data": {"text": "What is ML?", "is_query": true}, "model": "...", "task": "plugin"}
    Doc:   {"data": {"text": "ML is ...", "is_query": false}, "model": "...", "task": "plugin"}

Request format (offline):
    llm.encode({"data": {"text": "What is ML?", "is_query": true}})
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm_factory.io.base import (
    FactoryIOProcessor,
    PoolingRequestOutput,
    PromptType,
    TokensPrompt,
)

QUERY_PREFIX_ID = 50368  # [Q] with trailing space
DOC_PREFIX_ID = 50369  # [D] with trailing space


@dataclass
class ModernColBERTInput:
    """Validated embedding request after parse_request."""

    text: str
    is_query: bool = True


class ModernColBERTIOProcessor(FactoryIOProcessor):
    """IOProcessor for ModernColBERT — multi-vector late-interaction embeddings.

    Data flow:
        IOProcessorRequest(data={text, is_query})
        → factory_parse        → ModernColBERTInput
        → factory_pre_process  → TokensPrompt (with [Q]/[D] prefix at position 1)
        → merge_pooling_params → PoolingParams(task="plugin", extra_kwargs={...})
        → engine.encode        → PoolingRequestOutput
        → factory_post_process → base64-encoded flattened multi-vector embeddings
    """

    pooling_task = "token_embed"

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)

        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )

    def factory_parse(self, data: Any) -> ModernColBERTInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' key, got {type(data)}")

        if "text" not in data:
            raise ValueError("Request data must contain a 'text' key")

        is_query = bool(data.get("is_query", True))
        return ModernColBERTInput(text=data["text"], is_query=is_query)

    def factory_pre_process(
        self,
        parsed_input: ModernColBERTInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        is_query = parsed_input.is_query
        max_len = 256 if is_query else 8192
        prefix_id = QUERY_PREFIX_ID if is_query else DOC_PREFIX_ID

        tokens = self._tokenizer(
            parsed_input.text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len - 1,
            padding=False,
            return_tensors=None,
        )
        input_ids = [tokens["input_ids"][0], prefix_id] + tokens["input_ids"][1:]
        attention_mask = [1, 1] + tokens["attention_mask"][1:]

        extra = {
            "is_query": is_query,
            "sequence_length": len(input_ids),
            "attention_mask": attention_mask,
            "input_ids": input_ids,
        }

        self._stash(extra_kwargs=extra)

        return TokensPrompt(prompt_token_ids=input_ids)

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> str:
        import base64

        if not model_output:
            return ""

        output = model_output[0]
        raw = output.outputs.data
        if raw is None:
            return ""

        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw)

        return base64.b64encode(
            raw.cpu().contiguous().to(torch.float32).numpy().tobytes()
        ).decode("ascii")


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.moderncolbert.io_processor.ModernColBERTIOProcessor"
