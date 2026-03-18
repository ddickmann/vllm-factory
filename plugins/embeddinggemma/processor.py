"""
EmbeddingGemma Processor — sync inference via vLLM LLM.embed().

Mirrors the proven parity_test.py approach exactly:
  1. Tokenize with task-specific prompt prefix
  2. LLM.embed(TokensPrompt) → L2-normalized embeddings

Supported tasks/prompts (from config_sentence_transformers.json):
    query               → "task: search result | query: "
    document            → "title: none | text: "
    Clustering          → "task: clustering | query: "
    Classification      → "task: classification | query: "
    STS                 → "task: sentence similarity | query: "
    Summarization       → "task: summarization | query: "
    Retrieval-query     → "task: search result | query: "
    Retrieval-document  → "title: none | text: "
    ...and more
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm.inputs import TokensPrompt

logger = logging.getLogger("embeddinggemma.processor")

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


class EmbeddingGemmaProcessor:
    """Sync processor for EmbeddingGemma dense embeddings.

    Uses vLLM's LLM.embed() — the same path proven by parity_test.py.

    Usage:
        processor = EmbeddingGemmaProcessor("unsloth/embeddinggemma-300m")
        emb = processor.run("What is ML?", task="Clustering")
        embs = processor.run_batch(["q1", "q2"], task="STS")
    """

    def __init__(
        self,
        model_path: str,
        max_length: int = 2048,
        dtype: str = "float32",
        gpu_memory_utilization: float = 0.7,
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
    ):
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model_path = model_path
        self.max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=trust_remote_code,
        )

        self._llm = LLM(
            model=model_path,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
            dtype=dtype,
            enable_prefix_caching=False,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        logger.info("[EmbeddingGemmaProcessor] ✓ Engine ready")

    def _tokenize(self, text: str, task: str = "query") -> TokensPrompt:
        prompt_prefix = TASK_PROMPTS.get(task, TASK_PROMPTS["query"])
        full_text = prompt_prefix + text
        tokens = self._tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return TokensPrompt(prompt_token_ids=tokens["input_ids"])

    def run(self, text: str, task: str = "query") -> Optional[torch.Tensor]:
        """Embed a single text. Returns (dim,) tensor or None."""
        prompt = self._tokenize(text, task=task)
        outputs = self._llm.embed([prompt])
        if not outputs:
            return None
        return torch.as_tensor(outputs[0].outputs.embedding).float()

    def run_batch(
        self, texts: List[str], task: str = "query"
    ) -> List[Optional[torch.Tensor]]:
        """Embed a batch of texts. Returns list of (dim,) tensors."""
        if not texts:
            return []
        prompts = [self._tokenize(t, task=task) for t in texts]
        outputs = self._llm.embed(prompts)
        results: List[Optional[torch.Tensor]] = []
        for out in outputs:
            if out is not None and out.outputs is not None:
                results.append(
                    torch.as_tensor(out.outputs.embedding).float()
                )
            else:
                results.append(None)
        return results

    def close(self):
        """Release GPU memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
