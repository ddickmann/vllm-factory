"""
Plugin Processor Base — standardized async inference pipeline.

Every plugin processor follows the same 3-stage pattern:

    User Input → preprocess() → AsyncLLMEngine.encode() → postprocess() → Output

This module provides BaseProcessor with:
- Engine lifecycle: lazy init with asyncio.Lock, graceful shutdown
- preprocess(): Override to tokenize, add metadata, build PoolingParams
- postprocess(): Override to transform raw model output
- process_single(): preprocess → engine.encode → postprocess (with retry)
- process_batch(): concurrent asyncio.gather over process_single
- Sync wrappers: run() / run_batch() via asyncio.run()

Derived from patterns in:
- superpod/services/colbert/colbert_pipeline/processor.py
- superpod/services/compression/pipeline/processor.py

Example:
    class ColBERTProcessor(BaseProcessor):
        def preprocess(self, text, **kwargs):
            is_query = kwargs.get("is_query", True)
            tokens = self.tokenizer(text, ...)
            return PreprocessedInput(
                prompt=TokensPrompt(prompt_token_ids=tokens["input_ids"]),
                pooling_params=PoolingParams(task="token_embed", extra_kwargs={...}),
            )

        def postprocess(self, raw_output, metadata=None):
            return torch.as_tensor(raw_output)

    processor = ColBERTProcessor(model_path="my-model")
    embeddings = processor.run_batch(["query1", "query2"], is_query=True)
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("forge.processor")


@dataclass
class PreprocessedInput:
    """Output of preprocess() — everything needed for engine.encode().

    Attributes:
        prompt: Either a raw string or a TokensPrompt (pretokenized input).
        pooling_params: PoolingParams with task and optional extra_kwargs.
        metadata: Arbitrary dict passed through to postprocess() for context.
    """

    prompt: Any  # str | TokensPrompt
    pooling_params: Any  # PoolingParams
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseProcessor(ABC):
    """Abstract base class for plugin processors.

    Provides the full async inference pipeline:
        preprocess() → AsyncLLMEngine.encode() → postprocess()

    Subclasses MUST implement:
        preprocess(input_data, **kwargs) → PreprocessedInput
        postprocess(raw_output, metadata) → Any

    Subclasses MAY override:
        engine_kwargs() → dict   (extra AsyncEngineArgs)
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.7,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        enforce_eager: bool = True,
        quantization: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        """Initialize processor with engine configuration.

        The engine is NOT created here — it is created lazily on first use
        via _ensure_engine() to avoid blocking the constructor.

        Args:
            model_path: HuggingFace model ID or local path.
            dtype: Model dtype ("auto", "bfloat16", "float16", etc.).
            gpu_memory_utilization: Fraction of GPU memory to use.
            max_model_len: Maximum sequence length (auto-detected if None).
            max_num_seqs: Maximum concurrent sequences in the engine.
            enforce_eager: Disable CUDA graphs (recommended for encoder models).
            quantization: Quantization method (e.g., "fp8").
            trust_remote_code: Allow custom model code from HuggingFace.
        """
        # Prevent tokenizer deadlocks in async contexts
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model_path = model_path
        self._engine_config = {
            "model": model_path,
            "runner": "pooling",
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "trust_remote_code": trust_remote_code,
            "hf_token": os.getenv("HF_TOKEN"),
        }
        if max_model_len is not None:
            self._engine_config["max_model_len"] = max_model_len
            max_batched = kwargs.pop("max_num_batched_tokens", max_model_len)
            self._engine_config["max_num_batched_tokens"] = max_batched
        if quantization is not None:
            self._engine_config["quantization"] = quantization

        # Merge subclass overrides
        self._engine_config.update(self.engine_kwargs())

        self._engine = None
        self._engine_lock = asyncio.Lock()

    def engine_kwargs(self) -> dict:
        """Override to provide extra AsyncEngineArgs.

        Returns:
            Dict of additional keyword arguments merged into engine config.
        """
        return {}

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    async def _ensure_engine(self):
        """Lazily initialize the AsyncLLMEngine (thread-safe via lock)."""
        if self._engine is None:
            async with self._engine_lock:
                if self._engine is None:
                    from vllm.engine.arg_utils import AsyncEngineArgs
                    from vllm.engine.async_llm_engine import AsyncLLMEngine

                    logger.info(f"[{self.__class__.__name__}] Initializing engine...")
                    args = AsyncEngineArgs(**self._engine_config)
                    self._engine = AsyncLLMEngine.from_engine_args(args)
                    logger.info(f"[{self.__class__.__name__}] ✓ Engine ready")

    async def close(self):
        """Shutdown the engine and release GPU memory."""
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception:
                pass
            self._engine = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def preprocess(self, input_data: Any, **kwargs) -> PreprocessedInput:
        """Transform raw input into a PreprocessedInput for the engine.

        This is a SYNCHRONOUS method (CPU-bound tokenization).
        For truly async preprocessing, override preprocess_async() instead.

        Args:
            input_data: Raw input (typically a string or dict).
            **kwargs: Task-specific parameters (is_query, entity_labels, etc.).

        Returns:
            PreprocessedInput with prompt, pooling_params, and optional metadata.
        """
        ...

    @abstractmethod
    def postprocess(self, raw_output: Any, metadata: Optional[Dict] = None) -> Any:
        """Transform raw engine output into structured result.

        Args:
            raw_output: Raw tensor/data from engine.encode().
            metadata: The metadata dict from PreprocessedInput (passed through).

        Returns:
            Task-specific structured output.
        """
        ...

    # ------------------------------------------------------------------
    # Async preprocessing (override if tokenization must be async)
    # ------------------------------------------------------------------

    async def preprocess_async(self, input_data: Any, **kwargs) -> PreprocessedInput:
        """Async wrapper for preprocess(). Runs CPU-bound work in a thread.

        Override this directly if you need true async preprocessing.
        Default implementation dispatches to preprocess() via to_thread().
        """
        return await asyncio.to_thread(self.preprocess, input_data, **kwargs)

    # ------------------------------------------------------------------
    # Core processing methods
    # ------------------------------------------------------------------

    async def process_single(
        self,
        input_data: Any,
        max_retries: int = 3,
        **kwargs,
    ) -> Optional[Any]:
        """Process a single input through the full pipeline.

        preprocess → engine.encode → postprocess

        Args:
            input_data: Raw input to process.
            max_retries: Number of retry attempts on transient errors.
            **kwargs: Passed through to preprocess().

        Returns:
            Postprocessed output, or None on failure.
        """
        await self._ensure_engine()

        # Step 1: Preprocess
        prepared = await self.preprocess_async(input_data, **kwargs)

        # Step 2: Engine inference with retry
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(0.1 * attempt)

                request_id = str(uuid.uuid4())
                async for result in self._engine.encode(
                    prepared.prompt, prepared.pooling_params, request_id
                ):
                    raw_data = (
                        result.outputs.data
                        if hasattr(result.outputs, "data")
                        else None
                    )

                    # Step 3: Postprocess
                    return self.postprocess(raw_data, prepared.metadata)

                return None  # No result yielded

            except asyncio.CancelledError:
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                logger.warning(
                    f"[{self.__class__.__name__}] Attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt == max_retries - 1:
                    raise

        return None

    async def process_batch(
        self,
        inputs: List[Any],
        max_retries: int = 3,
        **kwargs,
    ) -> List[Optional[Any]]:
        """Process a batch of inputs concurrently.

        All inputs are submitted to the engine concurrently via asyncio.gather.
        vLLM's continuous batching automatically groups them for GPU efficiency.

        Args:
            inputs: List of raw inputs to process.
            max_retries: Number of retry attempts per input.
            **kwargs: Passed through to preprocess() for each input.

        Returns:
            List of postprocessed outputs (None for failed inputs).
        """
        await self._ensure_engine()

        if not inputs:
            return []

        results = await asyncio.gather(
            *[self.process_single(inp, max_retries=max_retries, **kwargs) for inp in inputs],
            return_exceptions=True,
        )

        # Replace exceptions with None
        return [
            r if not isinstance(r, BaseException) else None
            for r in results
        ]

    # ------------------------------------------------------------------
    # Sync wrappers (convenience for non-async contexts)
    # ------------------------------------------------------------------

    def run(self, input_data: Any, **kwargs) -> Optional[Any]:
        """Synchronous wrapper for process_single()."""
        return asyncio.run(self.process_single(input_data, **kwargs))

    def run_batch(self, inputs: List[Any], **kwargs) -> List[Optional[Any]]:
        """Synchronous wrapper for process_batch()."""
        return asyncio.run(self.process_batch(inputs, **kwargs))
