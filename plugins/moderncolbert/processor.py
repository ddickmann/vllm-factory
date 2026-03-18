"""
ModernColBERT Processor — async inference pipeline.

Preprocessing:  Tokenize + insert [Q]/[D] prefix at position 1
Engine:         AsyncLLMEngine.encode() with task="token_embed"
Postprocessing: Return raw multi-vector embeddings as torch.Tensor
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer

from forge.processor_base import BaseProcessor, PreprocessedInput

try:
    from vllm import PoolingParams
    from vllm.inputs import TokensPrompt
except ImportError:
    PoolingParams = None
    TokensPrompt = None


class ModernColBERTProcessor(BaseProcessor):
    """Async processor for ColBERT multi-vector embeddings.

    Usage:
        processor = ModernColBERTProcessor("answerdotai/ModernBERT-base")
        # Async
        embeddings = await processor.process_batch(["q1", "q2"], is_query=True)
        # Sync
        embeddings = processor.run_batch(["q1", "q2"], is_query=True)
    """

    # ModernBERT ColBERT special tokens
    QUERY_PREFIX_ID = 50368   # [Q] with trailing space
    DOC_PREFIX_ID = 50369     # [D] with trailing space

    def __init__(
        self,
        model_path: str,
        query_max_length: int = 256,
        document_max_length: int = 8192,
        **kwargs,
    ):
        super().__init__(model_path, **kwargs)
        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True,
        )

    def engine_kwargs(self) -> dict:
        return {"max_model_len": self.document_max_length}

    def preprocess(self, text: str, **kwargs) -> PreprocessedInput:
        is_query = kwargs.get("is_query", True)
        max_len = self.query_max_length if is_query else self.document_max_length
        prefix_id = self.QUERY_PREFIX_ID if is_query else self.DOC_PREFIX_ID

        tokens = self._tokenizer(
            text, add_special_tokens=True, truncation=True,
            max_length=max_len - 1, padding=False, return_tensors=None,
        )
        input_ids = [tokens["input_ids"][0], prefix_id] + tokens["input_ids"][1:]
        attention_mask = [1, 1] + tokens["attention_mask"][1:]

        extra = {
            "is_query": is_query,
            "sequence_length": len(input_ids),
            "attention_mask": attention_mask,
            "input_ids": input_ids,
        }

        return PreprocessedInput(
            prompt=TokensPrompt(prompt_token_ids=input_ids),
            pooling_params=PoolingParams(task="token_embed", extra_kwargs=extra),
            metadata={"is_query": is_query, "text": text},
        )

    def postprocess(self, raw_output: Any, metadata: Optional[Dict] = None) -> Optional[torch.Tensor]:
        if raw_output is None:
            return None
        return torch.as_tensor(raw_output)
