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
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
from vllm.inputs import TokensPrompt
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams

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


class GLiNERRerankIOProcessor(IOProcessor[GLiNERRerankInput, list[dict[str, Any]]]):
    """IOProcessor for modernbert_gliner_rerank — GLiNER uni-encoder reranker.

    Data flow:
        IOProcessorRequest(data={text, labels, ...})
        -> parse_request -> GLiNERRerankInput
        -> pre_process   -> TokensPrompt (+ stash extra_kwargs and metadata)
        -> validate_or_generate_params -> PoolingParams(extra_kwargs=gliner_data)
        -> engine.encode  -> PoolingRequestOutput
        -> post_process   -> list[dict] (decoded entities)
        -> output_to_response -> IOProcessorResponse(data=[...])
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

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

        self._lock = threading.Lock()
        self._pending_extra_kwargs: dict | None = None
        self._request_meta: dict[str, dict] = {}

        logger.info("GLiNERRerankIOProcessor initialized (uni-encoder path)")

    # ------------------------------------------------------------------
    # IOProcessor ABC implementation
    # ------------------------------------------------------------------

    def parse_request(self, request: Any) -> GLiNERRerankInput:
        if hasattr(request, "data"):
            data = request.data
        elif isinstance(request, dict) and "data" in request:
            data = request["data"]
        else:
            data = request

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dict with 'text' and 'labels' keys, got {type(data)}"
            )

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

    def pre_process(
        self,
        prompt: GLiNERRerankInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        words: list[str] = []
        word_starts: list[int] = []
        word_ends: list[int] = []
        for token, start, end in self._words_splitter(prompt.text):
            words.append(token)
            word_starts.append(start)
            word_ends.append(end)

        batch = self._collator(
            [{"tokenized_text": words, "ner": None}],
            entity_types=prompt.labels,
        )

        input_ids = batch["input_ids"][0].detach().cpu()
        words_mask = batch["words_mask"][0].detach().cpu()
        am = batch.get("attention_mask")
        if am is not None:
            attention_mask = am[0].detach().cpu()
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        tl = batch["text_lengths"]
        if tl.dim() == 2:
            text_length = int(tl[0, 0].item())
        else:
            text_length = int(tl[0].item())

        ids_list = input_ids.tolist()

        extra_kwargs = {
            "input_ids": ids_list,
            "attention_mask": attention_mask.tolist(),
            "words_mask": words_mask.tolist(),
            "text_lengths": text_length,
        }

        postprocess_meta = {
            "text": prompt.text,
            "words": words,
            "word_starts": word_starts,
            "word_ends": word_ends,
            "labels": prompt.labels,
            "threshold": prompt.threshold,
            "flat_ner": prompt.flat_ner,
            "multi_label": prompt.multi_label,
        }

        with self._lock:
            self._pending_extra_kwargs = extra_kwargs
            if request_id is not None:
                self._request_meta[request_id] = postprocess_meta
            else:
                self._request_meta["_offline"] = postprocess_meta

        return TokensPrompt(prompt_token_ids=ids_list)

    def validate_or_generate_params(
        self, params: PoolingParams | None = None,
    ) -> PoolingParams:
        with self._lock:
            extra = self._pending_extra_kwargs
            self._pending_extra_kwargs = None

        if params is not None and extra is not None:
            params.extra_kwargs = extra
            return params

        return PoolingParams(extra_kwargs=extra)

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        meta_key = request_id if request_id is not None else "_offline"
        with self._lock:
            meta = self._request_meta.pop(meta_key, None)

        if not model_output or meta is None:
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

        id_to_classes = {i + 1: label for i, label in enumerate(meta["labels"])}

        spans = self._decoder.decode(
            tokens=[meta["words"]],
            id_to_classes=id_to_classes,
            model_output=logits,
            flat_ner=meta.get("flat_ner", False),
            threshold=meta.get("threshold", 0.5),
            multi_label=meta.get("multi_label", False),
        )

        src = meta["text"]
        entities: list[dict[str, Any]] = []
        for span in spans[0]:
            ws = span.start
            we = span.end
            char_start = meta["word_starts"][ws] if ws < len(meta["word_starts"]) else 0
            char_end = meta["word_ends"][we] if we < len(meta["word_ends"]) else len(src)
            entities.append({
                "start": char_start,
                "end": char_end,
                "text": src[char_start:char_end],
                "label": span.entity_type,
                "score": round(span.score, 4),
            })

        return entities

    def output_to_response(
        self, plugin_output: list[dict[str, Any]],
    ) -> IOProcessorResponse:
        return IOProcessorResponse(data=plugin_output)


def get_processor_cls() -> str:
    """Entry-point callable for vllm.io_processor_plugins group."""
    return "plugins.modernbert_gliner_rerank.io_processor.GLiNERRerankIOProcessor"
