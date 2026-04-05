"""GenericPoolingIOProcessor — text-in / embedding-out for any backbone+pooler.

Handles the most common use-case: user sends text, gets back an embedding
vector (or multi-vector tensor).  Supports passing ``extra_kwargs`` through
to the pooler for advanced use-cases.

Entry-point group: ``vllm.io_processor_plugins``
Entry-point name:  ``generic_pooling_io``
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
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


class GenericPoolingIOProcessor(FactoryIOProcessor):
    """Generic IO processor for backbone + pooler serving.

    Request format (online POST /pooling)::

        {
            "data": {
                "text": "Hello world",
                "extra_kwargs": {"is_query": true}   // optional, passed to pooler
            },
            "model": "...",
            "task": "embed"
        }

    Request format (offline)::

        llm.encode({"data": {"text": "Hello world"}})

    Response: base64-encoded float32 numpy bytes of the embedding tensor.
    """

    pooling_task = "embed"

    def __init__(self, vllm_config: VllmConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )

    def factory_parse(self, data: Any) -> dict[str, Any]:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' key, got {type(data)}")
        if "text" not in data:
            raise ValueError("Request data must contain a 'text' key")

        return data

    def factory_pre_process(
        self,
        parsed_input: dict[str, Any],
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        text = parsed_input["text"]
        tokens = self._tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokens["input_ids"]

        extra = parsed_input.get("extra_kwargs") or {}
        extra["input_ids"] = input_ids

        self._stash(extra_kwargs=extra)

        return TokensPrompt(prompt_token_ids=input_ids)

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> str:
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
    """Entry-point callable for ``vllm.io_processor_plugins`` group."""
    return "vllm_factory.composable.io_processor.GenericPoolingIOProcessor"
