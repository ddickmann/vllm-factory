"""
IOProcessor plugin for embeddinggemma — dense embeddings via vLLM's native
IOProcessor pipeline with 13 task-specific prompt prefixes.

Handles text-only inputs, prepends a task-specific prefix, tokenizes with
max_length=2048, and returns a single dense embedding vector.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  embeddinggemma_io

Request format (online POST /pooling):
    {"data": {"text": "What is ML?", "task": "query"}, "model": "...", "task": "plugin"}
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

TASK_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "BitextMining": "task: search result | query: ",
    "Clustering": "task: clustering | query: ",
    "Classification": "task: classification | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-query": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "STS": "task: sentence similarity | query: ",
    "Summarization": "task: summarization | query: ",
}


@dataclass
class EmbeddingGemmaInput:
    """Validated embedding request after parse_request."""
    text: str
    task: str = "query"


class EmbeddingGemmaIOProcessor(IOProcessor[EmbeddingGemmaInput, list[float]]):
    """IOProcessor for EmbeddingGemma — unsloth/embeddinggemma-300m.

    Data flow:
        IOProcessorRequest(data={text, task?})
        → parse_request → EmbeddingGemmaInput
        → pre_process   → TokensPrompt(prompt_token_ids=...)
        → validate_or_generate_params → PoolingParams()
        → engine.embed   → PoolingRequestOutput
        → post_process   → list[float] (dense embedding)
        → output_to_response → IOProcessorResponse(data=[...])
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        from transformers import AutoTokenizer

        model_name = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True,
        )
        self.max_length = 2048

    def parse_request(self, request: Any) -> EmbeddingGemmaInput:
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

        task = data.get("task", "query")
        if task not in TASK_PROMPTS:
            raise ValueError(
                f"Unknown task '{task}'. Valid tasks: {list(TASK_PROMPTS.keys())}"
            )

        return EmbeddingGemmaInput(text=data["text"], task=task)

    def pre_process(
        self,
        prompt: EmbeddingGemmaInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        prefix = TASK_PROMPTS[prompt.task]
        full_text = prefix + prompt.text
        tokens = self._tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return TokensPrompt(prompt_token_ids=tokens["input_ids"])

    def validate_or_generate_params(
        self, params: PoolingParams | None = None,
    ) -> PoolingParams:
        if params is not None:
            return params
        return PoolingParams()

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
    return "plugins.embeddinggemma.io_processor.EmbeddingGemmaIOProcessor"
