"""
ColQwen3 Processor — async inference pipeline for multi-vector document embeddings.

Preprocessing:  Build vLLM multi-modal input (text or image)
Engine:         AsyncLLMEngine.encode() with task="token_embed"
Postprocessing: Return raw multi-vector embeddings as torch.Tensor
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from forge.processor_base import BaseProcessor, PreprocessedInput

try:
    from vllm import PoolingParams
except ImportError:
    PoolingParams = None


class ColQwen3Processor(BaseProcessor):
    """Async processor for ColQwen3 multi-vector embeddings.

    Supports both text and image inputs for document retrieval.

    Usage:
        processor = ColQwen3Processor("vidore/colqwen3-v1")

        # Text query
        emb = await processor.process_single("What is machine learning?")

        # Image document (path or URL)
        emb = await processor.process_single({"image": "/path/to/page.png"})
    """

    def __init__(self, model_path: str, **kwargs):
        kwargs.setdefault("max_model_len", 8192)
        super().__init__(model_path, **kwargs)

    def preprocess(self, input_data: Any, **kwargs) -> PreprocessedInput:
        is_query = kwargs.get("is_query", True)

        if isinstance(input_data, str):
            # Text input (query or text-only document)
            prompt = input_data
        elif isinstance(input_data, dict) and "image" in input_data:
            # Image input — pass through for vLLM's image processing
            prompt = input_data
        else:
            prompt = str(input_data)

        extra = {"is_query": is_query}

        return PreprocessedInput(
            prompt=prompt,
            pooling_params=PoolingParams(task="token_embed", extra_kwargs=extra),
            metadata={"is_query": is_query},
        )

    def postprocess(self, raw_output: Any, metadata: Optional[Dict] = None) -> Optional[torch.Tensor]:
        if raw_output is None:
            return None
        return torch.as_tensor(raw_output)
