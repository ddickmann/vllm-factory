"""
IOProcessor plugin for lfm2_colbert — LFM2-ColBERT multi-vector embeddings
via vLLM's native IOProcessor pipeline.

Handles text-only inputs, tokenized with max_length=512 and returned as
TokensPrompt for token-level embeddings.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  lfm2_colbert_io

Request format (online POST /pooling):
    {"data": {"text": "What is ML?"}, "model": "...", "task": "plugin"}
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
from vllm.inputs import TokensPrompt
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams


@dataclass
class LFM2ColBERTInput:
    """Validated embedding request after parse_request."""
    text: str


class LFM2ColBERTIOProcessor(IOProcessor[LFM2ColBERTInput, list[float]]):
    """IOProcessor for LFM2-ColBERT — LiquidAI/LFM2-ColBERT-350M.

    Data flow:
        IOProcessorRequest(data={text})
        → parse_request → LFM2ColBERTInput
        → pre_process   → TokensPrompt(prompt_token_ids=...)
        → validate_or_generate_params → PoolingParams(task="token_embed")
        → engine.encode  → PoolingRequestOutput
        → post_process   → list[float] (flattened multi-vector embeddings)
        → output_to_response → IOProcessorResponse(data=[...])
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        from transformers import AutoTokenizer

        model_name = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )

    def parse_request(self, request: Any) -> LFM2ColBERTInput:
        if hasattr(request, "data"):
            data = request.data
        elif isinstance(request, dict) and "data" in request:
            data = request["data"]
        else:
            data = request

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dict with 'text' key, got {type(data)}"
            )

        if "text" not in data:
            raise ValueError("Request data must contain a 'text' key")

        return LFM2ColBERTInput(text=data["text"])

    def pre_process(
        self,
        prompt: LFM2ColBERTInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        tokens = self._tokenizer(
            prompt.text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        return TokensPrompt(prompt_token_ids=tokens["input_ids"])

    def validate_or_generate_params(
        self, params: PoolingParams | None = None,
    ) -> PoolingParams:
        if params is not None:
            if params.task is None:
                params.task = "token_embed"
            return params
        return PoolingParams(task="token_embed")

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
    return "plugins.lfm2_colbert.io_processor.LFM2ColBERTIOProcessor"
