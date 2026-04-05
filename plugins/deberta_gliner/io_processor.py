"""
IOProcessor plugin for deberta_gliner — server-side GLiNER NER via vLLM's
native IOProcessor pipeline.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  deberta_gliner_io

Request format (online POST /pooling):
    {"data": {"text": "...", "labels": ["person", "org"], "threshold": 0.5},
     "model": "...", "task": "plugin"}

Request format (offline):
    llm.encode({"data": {"text": "...", "labels": [...]}})
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm_factory.io.base import FactoryIOProcessor, TokensPrompt, PromptType, PoolingRequestOutput

from forge.gliner_postprocessor import GLiNERDecoder, get_final_entities
from forge.gliner_preprocessor import GLiNERPreprocessor


@dataclass
class GLiNERInput:
    """Validated NER request after parse_request."""

    text: str
    labels: list[str]
    threshold: float = 0.5
    flat_ner: bool = False
    multi_label: bool = False


class DeBERTaGLiNERIOProcessor(FactoryIOProcessor):
    """IOProcessor for deberta_gliner — GLiNER NER with DeBERTa backbone.

    Data flow:
        IOProcessorRequest(data={text, labels, ...})
        → factory_parse   → GLiNERInput
        → factory_pre_process → TokensPrompt (+ stash extra_kwargs and metadata)
        → merge_pooling_params → PoolingParams(task="plugin", extra_kwargs=gliner_data)
        → engine.encode    → PoolingRequestOutput
        → factory_post_process → list[dict] (decoded entities)
    """

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        from plugins.deberta_gliner_linker.vllm_pooling_attention_mask import (
            apply_pooling_attention_mask_patch,
        )

        apply_pooling_attention_mask_patch()

        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        config = vllm_config.model_config.hf_config

        self._preprocessor = GLiNERPreprocessor(
            underlying_tokenizer=self._tokenizer,
            config=config,
            device="cpu",
            include_attention_mask=True,
        )
        self._decoder = GLiNERDecoder()

    # ------------------------------------------------------------------
    # FactoryIOProcessor implementation
    # ------------------------------------------------------------------

    def factory_parse(self, data: Any) -> GLiNERInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' and 'labels' keys, got {type(data)}")

        labels = data.get("labels", [])
        if not labels:
            raise ValueError("'labels' list must not be empty")

        return GLiNERInput(
            text=data.get("text", ""),
            labels=labels,
            threshold=float(data.get("threshold", 0.5)),
            flat_ner=bool(data.get("flat_ner", False)),
            multi_label=bool(data.get("multi_label", False)),
        )

    def factory_pre_process(
        self,
        parsed_input: GLiNERInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        result = self._preprocessor(parsed_input.text, parsed_input.labels, device="cpu")
        enc = result["model_inputs"]
        meta = result["postprocessing_metadata"]

        input_ids = enc["input_ids"][0]
        words_mask = enc["words_mask"][0]
        text_lengths = enc["text_lengths"][0].item()

        ids_list = input_ids.tolist()
        mask_list = words_mask.tolist()
        attn_list = enc["attention_mask"][0].tolist()

        gliner_data = {
            "input_ids": ids_list,
            "words_mask": mask_list,
            "text_lengths": text_lengths,
            "attention_mask": attn_list,
            "span_idx": enc["span_idx"][0].tolist(),
            "span_mask": enc["span_mask"][0].tolist(),
        }

        postprocess_meta = {
            "text": parsed_input.text,
            "labels": parsed_input.labels,
            "threshold": parsed_input.threshold,
            "flat_ner": parsed_input.flat_ner,
            "multi_label": parsed_input.multi_label,
            "tokens": meta["tokens"],
            "word_positions": meta["word_positions"],
            "id_to_classes": meta["id_to_classes"],
        }

        self._stash(extra_kwargs=gliner_data, request_id=request_id, meta=postprocess_meta)

        return TokensPrompt(prompt_token_ids=ids_list)

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> list[dict[str, Any]]:
        if not model_output or request_meta is None:
            return []

        output = model_output[0]
        raw = output.outputs.data
        if raw is None:
            return []

        scores = torch.as_tensor(raw) if not isinstance(raw, torch.Tensor) else raw

        if scores.dim() == 1 and scores.numel() > 3:
            L = int(scores[0].item())
            K = int(scores[1].item())
            C = int(scores[2].item())
            logits = scores[3:].reshape(1, L, K, C)
        elif scores.dim() == 3:
            logits = scores.unsqueeze(0)
        else:
            return []

        decoded = self._decoder.decode(
            tokens=request_meta["tokens"],
            id_to_classes=request_meta["id_to_classes"],
            logits=logits,
            flat_ner=request_meta.get("flat_ner", False),
            threshold=request_meta.get("threshold", 0.5),
            multi_label=request_meta.get("multi_label", False),
        )

        entities_batch = get_final_entities(
            decoded_outputs=decoded,
            word_positions=request_meta["word_positions"],
            original_texts=[request_meta["text"]],
        )

        return entities_batch[0] if entities_batch else []


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.deberta_gliner.io_processor.DeBERTaGLiNERIOProcessor"
