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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig

from vllm_factory.io.base import (
    FactoryIOProcessor,
    PoolingRequestOutput,
    PromptType,
)


@dataclass
class NemotronColEmbedInput:
    """Validated embedding request after parse_request."""

    text: str | None = None
    image: Any = None
    is_query: bool = True


class NemotronColEmbedIOProcessor(FactoryIOProcessor):
    """IOProcessor for NemotronColEmbed — nvidia/nemotron-colembed-vl-4b-v2.

    Data flow:
        IOProcessorRequest(data={text or image, is_query})
        → factory_parse        → NemotronColEmbedInput
        → factory_pre_process  → formatted prompt string (+ multi_modal_data for images)
        → merge_pooling_params → PoolingParams(task="plugin")
        → engine.encode        → PoolingRequestOutput
        → factory_post_process → base64-encoded flattened multi-vector embeddings
    """

    pooling_task = "token_embed"

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        from transformers import AutoProcessor

        model_name = vllm_config.model_config.model
        self._processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.query_prefix = "query: "
        self.passage_prefix = "passage: "

    def factory_parse(self, data: Any) -> NemotronColEmbedInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' or 'image' key, got {type(data)}")

        is_query = bool(data.get("is_query", True))

        if "text" in data:
            return NemotronColEmbedInput(text=data["text"], is_query=is_query)
        elif "image" in data:
            return NemotronColEmbedInput(image=data["image"], is_query=is_query)
        else:
            raise ValueError("Request data must contain either 'text' or 'image' key")

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

    def factory_pre_process(
        self,
        parsed_input: NemotronColEmbedInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        if parsed_input.text is not None:
            prefixed = f"{self.query_prefix}{parsed_input.text}"
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Query: {prefixed}"},
                    ],
                }
            ]
            formatted = self._processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted

        image = self._load_image(parsed_input.image)
        passage_text = f"{self.passage_prefix}"
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": passage_text},
                ],
            }
        ]
        formatted = self._processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt": formatted,
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

        return base64.b64encode(raw.cpu().contiguous().to(torch.float32).numpy().tobytes()).decode(
            "ascii"
        )


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.nemotron_colembed.io_processor.NemotronColEmbedIOProcessor"
