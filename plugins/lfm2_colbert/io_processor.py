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

from vllm_factory.io.base import (
    FactoryIOProcessor,
    PoolingRequestOutput,
    PromptType,
    TokensPrompt,
)


@dataclass
class LFM2ColBERTInput:
    """Validated embedding request after parse_request."""

    text: str


class LFM2ColBERTIOProcessor(FactoryIOProcessor):
    """IOProcessor for LFM2-ColBERT — LiquidAI/LFM2-ColBERT-350M.

    Data flow:
        IOProcessorRequest(data={text})
        → factory_parse        → LFM2ColBERTInput
        → factory_pre_process  → TokensPrompt(prompt_token_ids=...)
        → merge_pooling_params → PoolingParams(task="plugin")
        → engine.encode        → PoolingRequestOutput
        → factory_post_process → base64-encoded flattened multi-vector embeddings
    """

    pooling_task = "token_embed"

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        from transformers import AutoTokenizer

        model_name = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    def factory_parse(self, data: Any) -> LFM2ColBERTInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' key, got {type(data)}")

        if "text" not in data:
            raise ValueError("Request data must contain a 'text' key")

        return LFM2ColBERTInput(text=data["text"])

    def factory_pre_process(
        self,
        parsed_input: LFM2ColBERTInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        tokens = self._tokenizer(
            parsed_input.text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        return TokensPrompt(prompt_token_ids=tokens["input_ids"])

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

        return base64.b64encode(raw.cpu().contiguous().to(torch.float32).numpy().tobytes()).decode(
            "ascii"
        )


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.lfm2_colbert.io_processor.LFM2ColBERTIOProcessor"
