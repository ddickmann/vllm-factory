"""
IOProcessor plugin for nemotron_colembed — multimodal ColEmbed embeddings via
vLLM's native IOProcessor pipeline.

Handles text queries (prefixed with "query: ") and image document inputs
(prefixed with "passage: "), returning multi-vector embeddings as a list of
floats.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  nemotron_colembed_io

Request format (online POST /pooling):
    Text:  {"data": {"text": "What is ML?", "is_query": true}, "model": "...", "task": "plugin"}
    Image: {"data": {"image": "https://...", "is_query": false}, "model": "...", "task": "plugin"}
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams


@dataclass
class NemotronColEmbedInput:
    """Validated embedding request after parse_request."""
    text: str | None = None
    image: Any = None
    is_query: bool = True


class NemotronColEmbedIOProcessor(IOProcessor[NemotronColEmbedInput, list[float]]):
    """IOProcessor for NemotronColEmbed — nvidia/nemotron-colembed-vl-4b-v2.

    Data flow:
        IOProcessorRequest(data={text or image, is_query})
        → parse_request → NemotronColEmbedInput
        → pre_process   → formatted prompt string (+ multi_modal_data for images)
        → validate_or_generate_params → PoolingParams(task="token_embed")
        → engine.encode  → PoolingRequestOutput
        → post_process   → list[float] (flattened multi-vector embeddings)
        → output_to_response → IOProcessorResponse(data=[...])
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        from transformers import AutoProcessor

        model_name = vllm_config.model_config.model
        self._processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.query_prefix = "query: "
        self.passage_prefix = "passage: "

    def parse_request(self, request: Any) -> NemotronColEmbedInput:
        if hasattr(request, "data"):
            data = request.data
        elif isinstance(request, dict) and "data" in request:
            data = request["data"]
        else:
            data = request

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dict with 'text' or 'image' key, got {type(data)}"
            )

        is_query = bool(data.get("is_query", True))

        if "text" in data:
            return NemotronColEmbedInput(text=data["text"], is_query=is_query)
        elif "image" in data:
            return NemotronColEmbedInput(image=data["image"], is_query=is_query)
        else:
            raise ValueError(
                "Request data must contain either 'text' or 'image' key"
            )

    @staticmethod
    def _load_image(source):
        from PIL import Image as PILImage

        if isinstance(source, PILImage.Image):
            return source.convert("RGB")

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

    def pre_process(
        self,
        prompt: NemotronColEmbedInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        if prompt.text is not None:
            prefixed = f"{self.query_prefix}{prompt.text}"
            message = [{"role": "user", "content": [
                {"type": "text", "text": f"Query: {prefixed}"},
            ]}]
            formatted = self._processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True,
            )
            return formatted

        image = self._load_image(prompt.image)
        passage_text = f"{self.passage_prefix}"
        message = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": passage_text},
        ]}]
        formatted = self._processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True,
        )

        return {
            "prompt": formatted,
            "multi_modal_data": {"image": image},
        }

    def validate_or_generate_params(
        self, params: PoolingParams | None = None,
    ) -> PoolingParams:
        if params is not None:
            if params.task is None:
                params.task = "token_embed"
            return params
        return PoolingParams(task="token_embed")

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
    return "plugins.nemotron_colembed.io_processor.NemotronColEmbedIOProcessor"
