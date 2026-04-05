"""
IOProcessor plugin for deberta_gliner_linker — GLiNER entity linking via vLLM's
native IOProcessor pipeline (bi-encoder path).

Replaces the Forge BaseProcessor + pooling patch approach with vLLM's built-in
IOProcessor ABC.  The bi-encoder collator produces entity-type token prefixes and
the label embeddings are precomputed on CPU via GLiNER.encode_labels(), then
passed through PoolingParams.extra_kwargs alongside the collator attention mask.

Entry-point group: vllm.io_processor_plugins
Entry-point name:  deberta_gliner_linker_io

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

HF_MODEL_ID = "knowledgator/gliner-linker-large-v1.0"
_ENCODE_LABELS_BATCH_SIZE = 32


def _labels_cache_key(labels: list[str]) -> str:
    return "\x1f".join(labels)


def _cap_labels_tokenizer_max_length(gliner, max_length: int) -> None:
    """Cap the labels DeBERTa tokenizer model_max_length so encode_labels uses
    consistent fixed-width padding matching the GLinker L3 reference pipeline."""
    dp = getattr(gliner, "data_processor", None)
    if dp is None or not hasattr(dp, "labels_tokenizer"):
        return
    tok = dp.labels_tokenizer
    if getattr(tok, "model_max_length", 0) > 100_000:
        tok.model_max_length = max_length


@dataclass
class GLiNERLinkerInput:
    """Validated NER request after parse_request."""

    text: str
    labels: list[str]
    threshold: float = 0.5
    flat_ner: bool = False
    multi_label: bool = False


class GLiNERLinkerIOProcessor(FactoryIOProcessor):
    """IOProcessor for deberta_gliner_linker — GLiNER bi-encoder entity linker.

    Data flow:
        IOProcessorRequest(data={text, labels, ...})
        -> factory_parse   -> GLiNERLinkerInput
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
        from gliner.data_processing.collator import BiEncoderTokenDataCollator

        self._model_name = HF_MODEL_ID
        self._max_model_len = vllm_config.model_config.max_model_len

        gliner = GLiNER.from_pretrained(self._model_name)
        _cap_labels_tokenizer_max_length(gliner, self._max_model_len)

        dp = gliner.data_processor
        self._transformer_tokenizer = dp.transformer_tokenizer
        self._words_splitter = dp.words_splitter
        self._decoder = gliner.decoder
        self._config = gliner.config
        self._data_processor = dp
        self._collator = BiEncoderTokenDataCollator(
            gliner.config,
            data_processor=dp,
            return_tokens=True,
            return_id_to_classes=True,
            return_entities=True,
            prepare_labels=False,
        )

        self._cached_labels: list[str] | None = None
        self._label_embeddings_list: list | None = None
        self._warmed_label_keys: set[str] = set()

        del gliner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("GLiNERLinkerIOProcessor initialized (bi-encoder path)")

    # ------------------------------------------------------------------
    # Label encoding (cached; reloads GLiNER on CPU when labels change)
    # ------------------------------------------------------------------

    def _encode_labels(self, labels: list[str]) -> list:
        """Encode labels via GLiNER and return as nested Python list."""
        from gliner import GLiNER

        gliner = GLiNER.from_pretrained(self._model_name)
        _cap_labels_tokenizer_max_length(gliner, self._max_model_len)

        bs = min(_ENCODE_LABELS_BATCH_SIZE, max(1, len(labels)))
        embs = gliner.encode_labels(labels, batch_size=bs).cpu()

        del gliner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return embs.tolist()

    # ------------------------------------------------------------------
    # FactoryIOProcessor implementation
    # ------------------------------------------------------------------

    def factory_parse(self, data: Any) -> GLiNERLinkerInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'text' and 'labels' keys, got {type(data)}")

        labels = data.get("labels", [])
        if not labels:
            raise ValueError("'labels' list must not be empty")

        return GLiNERLinkerInput(
            text=data.get("text", ""),
            labels=labels,
            threshold=float(data.get("threshold", 0.5)),
            flat_ner=bool(data.get("flat_ner", False)),
            multi_label=bool(data.get("multi_label", False)),
        )

    def factory_pre_process(
        self,
        parsed_input: GLiNERLinkerInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        labels_key = _labels_cache_key(parsed_input.labels)
        if self._cached_labels != parsed_input.labels:
            self._label_embeddings_list = self._encode_labels(parsed_input.labels)
            self._cached_labels = list(parsed_input.labels)

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
        attention_mask = batch["attention_mask"][0].detach().cpu()
        words_mask = batch["words_mask"][0].detach().cpu()
        tl = batch["text_lengths"]
        if tl.dim() == 2:
            text_length = int(tl[0, 0].item())
        else:
            text_length = int(tl[0].item())

        ids_list = input_ids.tolist()

        extra_kwargs = {
            "attention_mask": attention_mask.tolist(),
            "words_mask": words_mask.tolist(),
            "text_lengths": text_length,
            "threshold": parsed_input.threshold,
            "labels_key": labels_key,
        }
        if labels_key not in self._warmed_label_keys:
            extra_kwargs["labels_embeds"] = self._label_embeddings_list

        postprocess_meta = {
            "text": parsed_input.text,
            "words": words,
            "word_starts": word_starts,
            "word_ends": word_ends,
            "labels": parsed_input.labels,
            "labels_key": labels_key,
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
        labels_key = request_meta.get("labels_key")
        if labels_key:
            self._warmed_label_keys.add(labels_key)

        output = model_output[0]
        raw = output.outputs.data
        if raw is None:
            return []

        scores = torch.as_tensor(raw) if not isinstance(raw, torch.Tensor) else raw

        span_logits = None
        span_idx = None
        span_mask = None

        if scores.dim() == 1 and scores.numel() > 3:
            scores = scores.to(dtype=torch.float32)
            W = int(scores[0].item())
            C = int(scores[1].item())
            S = int(scores[2].item())
            N = int(scores[3].item()) if scores.numel() > 4 else 0
            expected = 4 + (W * C * S) + (N * 2) + N + (N * C)
            if expected == scores.numel():
                offset = 4
                logits = scores[offset : offset + (W * C * S)].reshape(1, W, C, S)
                offset += W * C * S
                span_idx = scores[offset : offset + (N * 2)].reshape(1, N, 2).long()
                offset += N * 2
                span_mask = scores[offset : offset + N].reshape(1, N).bool()
                offset += N
                span_logits = scores[offset : offset + (N * C)].reshape(1, N, C)
            else:
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
            span_logits=span_logits,
            span_idx=span_idx,
            span_mask=span_mask,
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
    return "plugins.deberta_gliner_linker.io_processor.GLiNERLinkerIOProcessor"
