"""
IOProcessor plugin for modernbert_gliner_rerank — GLiNER L4 reranker via vLLM's
native IOProcessor pipeline (uni-encoder path).

Replaces the Forge BaseProcessor + pooling patch approach with vLLM's built-in
IOProcessor ABC.  The uni-encoder collator embeds entity-type labels inline in the
prompt (``[ENT, type, …, SEP, text tokens]``), so label embeddings are NOT
precomputed — the pooler reads them from hidden states via extract_prompt_features.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  modernbert_gliner_rerank_io

Request format (online POST /pooling):
    {"data": {"text": "...", "labels": ["person", "org"], "threshold": 0.5},
     "model": "...", "task": "plugin"}

Request format (offline):
    llm.encode({"data": {"text": "...", "labels": [...]}})
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm_factory.io.base import FactoryIOProcessor, TokensPrompt, PromptType, PoolingRequestOutput

logger = logging.getLogger(__name__)

HF_MODEL_ID = "knowledgator/gliner-linker-rerank-v1.0"


@dataclass
class GLiNERRerankInput:
    """Validated NER request after parse_request."""

    text: str
    labels: list[str]
    threshold: float = 0.5
    flat_ner: bool = False
    multi_label: bool = False


class GLiNERRerankIOProcessor(FactoryIOProcessor):
    """IOProcessor for modernbert_gliner_rerank — GLiNER uni-encoder reranker.

    Data flow:
        IOProcessorRequest(data={text, labels, ...})
        -> factory_parse   -> GLiNERRerankInput
        -> factory_pre_process -> TokensPrompt (+ stash extra_kwargs and metadata)
        -> merge_pooling_params -> PoolingParams(task="plugin", extra_kwargs=gliner_data)
        -> engine.encode    -> PoolingRequestOutput
        -> factory_post_process -> list[dict] (decoded entities)
    """

    pooling_task = "plugin"

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)

        from plugins.deberta_gliner_linker.vllm_pooling_attention_mask import (
            apply_pooling_attention_mask_patch,
        )

        apply_pooling_attention_mask_patch()

        from gliner import GLiNER
        from gliner.data_processing.collator import TokenDataCollator

        gliner = GLiNER.from_pretrained(HF_MODEL_ID)

        dp = gliner.data_processor
        self._transformer_tokenizer = dp.transformer_tokenizer
        self._words_splitter = dp.words_splitter
        self._decoder = gliner.decoder
        self._config = gliner.config
        self._data_processor = dp
        self._collator = TokenDataCollator(
            gliner.config,
            data_processor=dp,
            return_tokens=True,
            return_id_to_classes=True,
            prepare_labels=False,
        )

        del gliner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("GLiNERRerankIOProcessor initialized (uni-encoder path)")

    # ------------------------------------------------------------------
    # FactoryIOProcessor implementation
    # ------------------------------------------------------------------

    def factory_parse(self, data: Any) -> GLiNERRerankInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' and 'labels' keys, got {type(data)}")

        labels = data.get("labels", [])
        if not labels:
            raise ValueError("'labels' list must not be empty")

        return GLiNERRerankInput(
            text=data.get("text", ""),
            labels=labels,
            threshold=float(data.get("threshold", 0.5)),
            flat_ner=bool(data.get("flat_ner", False)),
            multi_label=bool(data.get("multi_label", False)),
        )

    def factory_pre_process(
        self,
        parsed_input: GLiNERRerankInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        words: list[str] = []
        word_starts: list[int] = []
        word_ends: list[int] = []
        for token, start, end in self._words_splitter(parsed_input.text):
            words.append(token)
            word_starts.append(start)
            word_ends.append(end)

        batch = self._collator(
            [{"tokenized_text": words, "ner": None}],
            entity_types=parsed_input.labels,
        )

        input_ids = batch["input_ids"][0].detach().cpu()
        words_mask = batch["words_mask"][0].detach().cpu()
        tl = batch["text_lengths"]
        if tl.dim() == 2:
            text_length = int(tl[0, 0].item())
        else:
            text_length = int(tl[0].item())

        ids_list = input_ids.tolist()

        extra_kwargs = {
            "words_mask": words_mask.tolist(),
            "text_lengths": text_length,
        }

        postprocess_meta = {
            "text": parsed_input.text,
            "words": words,
            "word_starts": word_starts,
            "word_ends": word_ends,
            "labels": parsed_input.labels,
            "threshold": parsed_input.threshold,
            "flat_ner": parsed_input.flat_ner,
            "multi_label": parsed_input.multi_label,
        }

        self._stash(extra_kwargs=extra_kwargs, request_id=request_id, meta=postprocess_meta)

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
            W = int(scores[0].item())
            C = int(scores[1].item())
            S = int(scores[2].item())
            logits = scores[3:].reshape(1, W, C, S)
        elif scores.dim() == 3:
            logits = scores.unsqueeze(0)
        else:
            return []

        id_to_classes = {i + 1: label for i, label in enumerate(request_meta["labels"])}

        spans = self._decoder.decode(
            tokens=[request_meta["words"]],
            id_to_classes=id_to_classes,
            model_output=logits,
            flat_ner=request_meta.get("flat_ner", False),
            threshold=request_meta.get("threshold", 0.5),
            multi_label=request_meta.get("multi_label", False),
        )

        src = request_meta["text"]
        entities: list[dict[str, Any]] = []
        for span in spans[0]:
            ws = span.start
            we = span.end
            char_start = request_meta["word_starts"][ws] if ws < len(request_meta["word_starts"]) else 0
            char_end = request_meta["word_ends"][we] if we < len(request_meta["word_ends"]) else len(src)
            entities.append(
                {
                    "start": char_start,
                    "end": char_end,
                    "text": src[char_start:char_end],
                    "label": span.entity_type,
                    "score": round(span.score, 4),
                }
            )

        return entities


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.modernbert_gliner_rerank.io_processor.GLiNERRerankIOProcessor"
