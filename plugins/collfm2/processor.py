"""
ColLFM2 Processor — async inference pipeline for multi-vector document embeddings.

Mirrors the reference preprocessing from:
    superpod/services/colpali/colpali_pipeline/collfm2_processor.py
    superpod/services/colpali/colpali_pipeline/processor.py

Preprocessing rules (EXACT match with superpod ColLFM2Processor):
    Queries: Direct tokenization, NO prefix, padding="longest", max_length=2048
             Passed as pre-tokenized {"prompt_token_ids": ids} to avoid any
             re-tokenization by vLLM's fast tokenizer.
    Images:  VISUAL_PROMPT_PREFIX text + multi_modal_data={"image": PIL}
             vLLM handles image expansion and encoding.

Postprocessing:
    Queries: Return raw embedding tensor as-is.
    Images:  Strip leading BOS token (vLLM prepends <|startoftext|>, reference doesn't).

Engine kwargs:
    skip_mm_profiling=True     — prevents OOM during VLM profiling
    mm_processor_cache_gb=1    — reduces multimodal cache from default 4GB
    limit_mm_per_prompt=1      — one image per request (ColPali constraint)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import torch

from forge.processor_base import BaseProcessor, PreprocessedInput

try:
    from vllm import PoolingParams
    from vllm.inputs import TokensPrompt
except ImportError:
    PoolingParams = None
    TokensPrompt = None


# Exact visual prompt prefix from superpod ColLFM2Processor (reference)
_VISUAL_PROMPT_PREFIX = "<|im_start|>user\n<image>Describe the image.<|im_end|>"


class ColLFM2Processor(BaseProcessor):
    """Async processor for ColLFM2 multi-vector embeddings.

    Produces (num_tokens, 128) L2-normalized embeddings per input for
    late-interaction MaxSim retrieval (same as ColQwen3 but ~6x smaller).

    Usage:
        processor = ColLFM2Processor("VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1")

        # Query
        emb = await processor.process_single("What is machine learning?", is_query=True)

        # Image document (PIL Image or path string)
        from PIL import Image
        emb = await processor.process_single(Image.open("page.png"), is_query=False)

        # Sync convenience
        emb = processor.run("What is machine learning?", is_query=True)
    """

    # Visual prompt prefix — MUST match superpod ColLFM2Processor.VISUAL_PROMPT_PREFIX
    VISUAL_PROMPT_PREFIX = _VISUAL_PROMPT_PREFIX

    def __init__(self, model_path: str, **kwargs):
        # sane defaults for the ~0.9GB model
        kwargs.setdefault("max_model_len", 8192)
        kwargs.setdefault("gpu_memory_utilization", 0.7)
        kwargs.setdefault("enforce_eager", True)
        super().__init__(model_path, **kwargs)
        self._hf_tokenizer = None
        self._tokenizer_lock = asyncio.Lock()

    def engine_kwargs(self) -> dict:
        """Extra AsyncEngineArgs required for LFM2-VL pooling.

        Mirrors AsyncEngineArgs in superpod/services/colpali/colpali_pipeline/processor.py.
        Without skip_mm_profiling the tensor-parallel profiler OOMs on first load.
        """
        return {
            "skip_mm_profiling": True,       # avoids OOM during VLM memory profiling
            "mm_processor_cache_gb": 1,       # keep multimodal cache small (default: 4GB)
            "limit_mm_per_prompt": {"image": 1},  # ColPali: one image per request
            "enable_prefix_caching": False,   # not compatible with pooling runner
            "enable_chunked_prefill": False,  # not compatible with encoder models
        }

    # ------------------------------------------------------------------
    # Lazy tokenizer — loaded once, alongside engine
    # ------------------------------------------------------------------

    async def _ensure_tokenizer(self):
        """Load HF tokenizer lazily (only once) for query preprocessing."""
        if self._hf_tokenizer is None:
            async with self._tokenizer_lock:
                if self._hf_tokenizer is None:
                    from transformers import AutoProcessor
                    proc = await asyncio.to_thread(
                        AutoProcessor.from_pretrained,
                        self.model_path,
                        trust_remote_code=True,
                    )
                    # Use underlying tokenizer for query tokenization
                    self._hf_tokenizer = proc.tokenizer if hasattr(proc, "tokenizer") else proc

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, input_data: Any, **kwargs) -> PreprocessedInput:
        """Synchronous preprocessing — dispatched via to_thread() by base class.

        For queries: pre-tokenizes without any prefix (mirrors reference).
        For images:  uses VISUAL_PROMPT_PREFIX text + multimodal dict.
        """
        is_query = kwargs.get("is_query", not self._is_image(input_data))

        if is_query:
            return self._preprocess_query_sync(input_data)
        else:
            return self._preprocess_image_sync(input_data)

    async def preprocess_async(self, input_data: Any, **kwargs) -> PreprocessedInput:
        """Async preprocessing — ensures tokenizer is loaded before sync call."""
        await self._ensure_engine()
        is_query = kwargs.get("is_query", not self._is_image(input_data))
        if is_query:
            await self._ensure_tokenizer()
        return await asyncio.to_thread(self.preprocess, input_data, **kwargs)

    def _is_image(self, input_data: Any) -> bool:
        """Detect whether input is an image (PIL or path)."""
        try:
            from PIL import Image as PILImage
            if isinstance(input_data, PILImage.Image):
                return True
        except ImportError:
            pass
        if isinstance(input_data, dict) and "image" in input_data:
            return True
        return False

    def _preprocess_query_sync(self, text: str) -> PreprocessedInput:
        """
        Query preprocessing — EXACT match with superpod ColLFM2Processor.process_queries():
        - Direct tokenization, NO prefix
        - padding="longest", max_length=2048
        - Returns pre-tokenized prompt_token_ids (bypasses vLLM's fast tokenizer)
        """
        if self._hf_tokenizer is None:
            # Fallback: load synchronously if called outside async context
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            tokenizer = proc.tokenizer if hasattr(proc, "tokenizer") else proc
        else:
            tokenizer = self._hf_tokenizer

        text = str(text)
        batch = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
            max_length=2048,
            truncation=True,
        )
        input_ids = batch["input_ids"][0].tolist()

        return PreprocessedInput(
            prompt=TokensPrompt(prompt_token_ids=input_ids),
            pooling_params=PoolingParams(task="token_embed"),
            metadata={"is_query": True, "text": text},
        )

    def _preprocess_image_sync(self, input_data: Any) -> PreprocessedInput:
        """
        Image preprocessing — EXACT match with superpod ColLFM2Processor.process_images():
        - text = VISUAL_PROMPT_PREFIX
        - multi_modal_data = {"image": PIL.Image}
        - vLLM handles pixel encoding internally
        """
        from PIL import Image as PILImage

        # resolve image
        if isinstance(input_data, PILImage.Image):
            image = input_data
        elif isinstance(input_data, dict) and "image" in input_data:
            val = input_data["image"]
            if isinstance(val, str):
                image = PILImage.open(val)
            else:
                image = val
        elif isinstance(input_data, str):
            image = PILImage.open(input_data)
        else:
            raise ValueError(f"Unsupported image input type: {type(input_data)}")

        image = image.convert("RGB")

        return PreprocessedInput(
            prompt={
                "prompt": self.VISUAL_PROMPT_PREFIX,
                "multi_modal_data": {"image": image},
            },
            pooling_params=PoolingParams(task="token_embed"),
            metadata={"is_query": False},
        )

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def postprocess(self, raw_output: Any, metadata: Optional[Dict] = None) -> Optional[torch.Tensor]:
        """
        For images: strip leading BOS token added by vLLM.
        vLLM prepends <|startoftext|> to image sequences; the HF reference doesn't.
        This ensures exact parity with the reference implementation.

        For queries: return raw embedding as-is.
        """
        if raw_output is None:
            return None

        emb = torch.as_tensor(raw_output)

        is_image = (metadata or {}).get("is_query", True) is False
        if is_image and emb.shape[0] > 1:
            # Strip BOS token (first position) to match reference output
            emb = emb[1:]

        return emb
