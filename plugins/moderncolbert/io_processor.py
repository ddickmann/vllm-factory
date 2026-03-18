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

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer

from vllm.config import VllmConfig
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
from vllm.inputs import TokensPrompt
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams


QUERY_PREFIX_ID = 50368   # [Q] with trailing space
DOC_PREFIX_ID = 50369     # [D] with trailing space


@dataclass
class ModernColBERTInput:
    """Validated embedding request after parse_request."""
    text: str
    is_query: bool = True


class ModernColBERTIOProcessor(IOProcessor[ModernColBERTInput, list[float]]):
    """IOProcessor for ModernColBERT — multi-vector late-interaction embeddings.

    Data flow:
        IOProcessorRequest(data={text, is_query})
        → parse_request → ModernColBERTInput
        → pre_process   → TokensPrompt (with [Q]/[D] prefix at position 1)
        → validate_or_generate_params → PoolingParams(task="token_embed", extra_kwargs={...})
        → engine.encode  → PoolingRequestOutput
        → post_process   → list[float] (flattened multi-vector embeddings)
        → output_to_response → IOProcessorResponse(data=[...])
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self._lock = threading.Lock()
        self._pending_extra_kwargs: dict | None = None

        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True,
        )

    def parse_request(self, request: Any) -> ModernColBERTInput:
        if hasattr(request, "data"):
            data = request.data
        elif isinstance(request, dict) and "data" in request:
            data = request["data"]
        else:
            data = request

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' key, got {type(data)}")

        if "text" not in data:
            raise ValueError("Request data must contain a 'text' key")

        is_query = bool(data.get("is_query", True))
        return ModernColBERTInput(text=data["text"], is_query=is_query)

    def pre_process(
        self,
        prompt: ModernColBERTInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        is_query = prompt.is_query
        max_len = 256 if is_query else 8192
        prefix_id = QUERY_PREFIX_ID if is_query else DOC_PREFIX_ID

        tokens = self._tokenizer(
            prompt.text,
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

        with self._lock:
            self._pending_extra_kwargs = extra

        return TokensPrompt(prompt_token_ids=input_ids)

    def validate_or_generate_params(
        self, params: PoolingParams | None = None,
    ) -> PoolingParams:
        with self._lock:
            extra = self._pending_extra_kwargs
            self._pending_extra_kwargs = None

        if params is not None and extra is not None:
            params.extra_kwargs = extra
            if params.task is None:
                params.task = "token_embed"
            return params

        return PoolingParams(task="token_embed", extra_kwargs=extra)

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> list[float]:
        if not model_output:
            return []

        output = model_output[0]
        raw = output.outputs.data
        if raw is None:
            return []

        if isinstance(raw, torch.Tensor):
            return raw.tolist()
        elif isinstance(raw, list):
            return raw
        else:
            return torch.as_tensor(raw).tolist()

    def output_to_response(
        self, plugin_output: list[float],
    ) -> IOProcessorResponse:
        return IOProcessorResponse(data=plugin_output)


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.moderncolbert.io_processor.ModernColBERTIOProcessor"
