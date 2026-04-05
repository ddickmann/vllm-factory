"""
IOProcessor plugin for collfm2 — ColPali multi-vector embeddings via vLLM's
native IOProcessor pipeline.

Handles text queries (direct tokenization, no prefix) and image document
inputs (visual prompt + multimodal data), returning multi-vector embeddings
as a list of floats.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  collfm2_io

Request format (online POST /pooling):
    Text:  {"data": {"text": "What is ML?", "is_query": true}, "model": "...", "task": "plugin"}
    Image: {"data": {"image": "https://..."}, "model": "...", "task": "plugin"}

Request format (offline):
    llm.encode({"data": {"text": "What is ML?", "is_query": true}})
    llm.encode({"data": {"image": "path/to/image.png"}})
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config import VllmConfig

from vllm_factory.io.base import (
    FactoryIOProcessor,
    PoolingRequestOutput,
    PromptType,
    TokensPrompt,
)

VISUAL_PROMPT_PREFIX = "<|im_start|>user\n<image>Describe the image.<|im_end|>"


@dataclass
class ColLFM2Input:
    """Validated embedding request after parse_request."""

    prompt: str | dict
    is_query: bool = True
    metadata: dict = field(default_factory=dict)


class ColLFM2IOProcessor(FactoryIOProcessor):
    """IOProcessor for ColLFM2 — late-interaction multi-vector embeddings.

    Data flow:
        IOProcessorRequest(data={text or image, is_query})
        → factory_parse        → ColLFM2Input
        → factory_pre_process  → TokensPrompt (queries) or dict prompt (images)
        → merge_pooling_params → PoolingParams(task="plugin")
        → engine.encode        → PoolingRequestOutput
        → factory_post_process → base64-encoded flattened multi-vector embeddings
    """

    pooling_task = "token_embed"

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)

        self._model_id = vllm_config.model_config.model
        self._hf_tokenizer = None
        self._tokenizer_lock = threading.Lock()

    def _ensure_tokenizer(self):
        """Load HF tokenizer lazily (thread-safe, once)."""
        if self._hf_tokenizer is None:
            with self._tokenizer_lock:
                if self._hf_tokenizer is None:
                    from transformers import AutoProcessor

                    proc = AutoProcessor.from_pretrained(
                        self._model_id,
                        trust_remote_code=True,
                    )
                    self._hf_tokenizer = proc.tokenizer if hasattr(proc, "tokenizer") else proc

    def factory_parse(self, data: Any) -> ColLFM2Input:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' or 'image' key, got {type(data)}")

        if "text" in data:
            is_query = bool(data.get("is_query", True))
            return ColLFM2Input(
                prompt=data["text"],
                is_query=is_query,
                metadata={"is_query": is_query},
            )
        elif "image" in data:
            is_query = bool(data.get("is_query", False))
            return ColLFM2Input(
                prompt={"image": data["image"]},
                is_query=is_query,
                metadata={"is_query": is_query},
            )
        else:
            raise ValueError("Request data must contain either 'text' or 'image' key")

    def _load_image(self, source: Any):
        """Resolve an image source to a PIL Image."""
        from PIL import Image as PILImage

        if isinstance(source, PILImage.Image):
            return source.convert("RGB")

        if isinstance(source, dict) and "image" in source:
            source = source["image"]

        if isinstance(source, str):
            if source.startswith("data:"):
                import base64
                from io import BytesIO

                header, b64data = source.split(",", 1)
                return PILImage.open(BytesIO(base64.b64decode(b64data))).convert("RGB")
            if source.startswith(("http://", "https://")):
                import urllib.request
                from io import BytesIO

                with urllib.request.urlopen(source) as resp:
                    return PILImage.open(BytesIO(resp.read())).convert("RGB")
            return PILImage.open(source).convert("RGB")

        raise ValueError(f"Unsupported image source type: {type(source)}")

    def factory_pre_process(
        self,
        parsed_input: ColLFM2Input,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        metadata = parsed_input.metadata

        if parsed_input.is_query:
            self._ensure_tokenizer()
            batch = self._hf_tokenizer(
                str(parsed_input.prompt),
                return_tensors="pt",
                padding="longest",
                return_attention_mask=True,
                max_length=2048,
                truncation=True,
            )
            input_ids = batch["input_ids"][0].tolist()

            self._stash(extra_kwargs={}, request_id=request_id, meta=metadata)

            return TokensPrompt(prompt_token_ids=input_ids)
        else:
            image = self._load_image(parsed_input.prompt)

            self._stash(extra_kwargs={}, request_id=request_id, meta=metadata)

            return {
                "prompt": VISUAL_PROMPT_PREFIX,
                "multi_modal_data": {"image": image},
            }

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

        if isinstance(raw, torch.Tensor):
            emb = raw
        else:
            emb = torch.as_tensor(raw)

        is_image = (request_meta or {}).get("is_query", True) is False
        if is_image and emb.shape[0] > 1:
            emb = emb[1:]

        return base64.b64encode(emb.cpu().contiguous().to(torch.float32).numpy().tobytes()).decode(
            "ascii"
        )


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.collfm2.io_processor.ColLFM2IOProcessor"
