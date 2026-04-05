"""
IOProcessor plugin for colqwen3 — multimodal ColPali embeddings via vLLM's
native IOProcessor pipeline.

Handles text queries and image document inputs, returning multi-vector
embeddings as a list of floats.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  colqwen3_io

Request format (online POST /pooling):
    Text:  {"data": {"text": "What is ML?"}, "model": "...", "task": "plugin"}
    Image: {"data": {"image": "https://..."}, "model": "...", "task": "plugin"}

Request format (offline):
    llm.encode({"data": {"text": "What is ML?", "is_query": true}})
"""

from __future__ import annotations

import threading
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
class ColQwen3Input:
    """Validated embedding request after parse_request."""

    prompt: str | dict
    is_query: bool = True


QUERY_PREFIX = "Query: "
QUERY_AUG_TOKEN = "<|endoftext|>"
QUERY_AUG_SUFFIX = QUERY_AUG_TOKEN * 10
VISUAL_PROMPT_PREFIX = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe the image.<|im_end|><|endoftext|>"
)


class ColQwen3IOProcessor(FactoryIOProcessor):
    """IOProcessor for ColQwen3 — Qwen3-VL + ColPali multi-vector embeddings.

    Data flow:
        IOProcessorRequest(data={text or image, is_query})
        → factory_parse        → ColQwen3Input
        → factory_pre_process  → TokensPrompt (queries) or dict prompt (images)
        → merge_pooling_params → PoolingParams(task="plugin", extra_kwargs={is_query})
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
        if self._hf_tokenizer is None:
            with self._tokenizer_lock:
                if self._hf_tokenizer is None:
                    from transformers import AutoTokenizer

                    self._hf_tokenizer = AutoTokenizer.from_pretrained(
                        self._model_id,
                        trust_remote_code=True,
                    )

    def factory_parse(self, data: Any) -> ColQwen3Input:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' or 'image' key, got {type(data)}")

        is_query = bool(data.get("is_query", True))

        if "text" in data:
            return ColQwen3Input(prompt=data["text"], is_query=is_query)
        elif "image" in data:
            return ColQwen3Input(prompt={"image": data["image"]}, is_query=is_query)
        else:
            raise ValueError("Request data must contain either 'text' or 'image' key")

    @staticmethod
    def _load_image(source: Any):
        from PIL import Image as PILImage

        if isinstance(source, PILImage.Image):
            return source.convert("RGB")

        if isinstance(source, dict) and "image" in source:
            source = source["image"]

        if isinstance(source, str):
            if source.startswith("data:"):
                import base64
                from io import BytesIO

                _, b64data = source.split(",", 1)
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
        parsed_input: ColQwen3Input,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        self._stash(extra_kwargs={"is_query": parsed_input.is_query})

        if isinstance(parsed_input.prompt, str):
            self._ensure_tokenizer()
            text = QUERY_PREFIX + parsed_input.prompt + QUERY_AUG_SUFFIX
            ids = self._hf_tokenizer(text, return_tensors="pt").input_ids[0].tolist()
            return TokensPrompt(prompt_token_ids=ids)

        image = self._load_image(parsed_input.prompt)
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

        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw)

        return base64.b64encode(
            raw.cpu().contiguous().to(torch.float32).numpy().tobytes()
        ).decode("ascii")


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.colqwen3.io_processor.ColQwen3IOProcessor"
