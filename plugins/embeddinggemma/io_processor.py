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
from vllm_factory.io.base import (
    FactoryIOProcessor,
    PoolingRequestOutput,
    PromptType,
    TokensPrompt,
)

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


class EmbeddingGemmaIOProcessor(FactoryIOProcessor):
    """IOProcessor for EmbeddingGemma — unsloth/embeddinggemma-300m.

    Data flow:
        IOProcessorRequest(data={text, task?})
        → factory_parse        → EmbeddingGemmaInput
        → factory_pre_process  → TokensPrompt(prompt_token_ids=...)
        → merge_pooling_params → PoolingParams(task="plugin")
        → engine.embed         → PoolingRequestOutput
        → factory_post_process → list[float] (dense embedding)
    """

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        from transformers import AutoTokenizer

        model_name = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        self.max_length = 2048

    def factory_parse(self, data: Any) -> EmbeddingGemmaInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' key, got {type(data)}")

        if "text" not in data:
            raise ValueError("Request data must contain a 'text' key")

        task = data.get("task", "query")
        if task not in TASK_PROMPTS:
            raise ValueError(f"Unknown task '{task}'. Valid tasks: {list(TASK_PROMPTS.keys())}")

        return EmbeddingGemmaInput(text=data["text"], task=task)

    def factory_pre_process(
        self,
        parsed_input: EmbeddingGemmaInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        prefix = TASK_PROMPTS[parsed_input.task]
        full_text = prefix + parsed_input.text
        tokens = self._tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return TokensPrompt(prompt_token_ids=tokens["input_ids"])

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
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


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.embeddinggemma.io_processor.EmbeddingGemmaIOProcessor"
