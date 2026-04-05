"""
GLiNER L4 reranker processor — same GLiNER preprocessing stack as vanilla inference, vLLM for the backbone.

Uses GLinker's uni-encoder path (not the linker's bi-encoder):

  - ``WordsSplitter`` + ``UniEncoderTokenProcessor`` + ``TokenDataCollator``
  - ``entity_types=labels`` builds the ``[ENT, type, …]`` prompt prefix; label vectors are **not**
    precomputed (the pooler reads them from hidden states via ``extract_prompt_features``).
  - ``TokenDecoder`` + threshold / ``flat_ner`` / ``multi_label`` for BIO decoding

Lifecycle::

    proc = GLiNERRerankProcessor()
    proc.warmup(labels)
    entities = proc.predict_entities("Apple is in California.", threshold=0.5)
    proc.close()
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

HF_MODEL_ID = "knowledgator/gliner-linker-rerank-v1.0"


class GLiNERRerankProcessor:
    """Entity extraction with GLiNER L4-style tensors and vLLM for ModernBERT + projection."""

    def __init__(
        self,
        model_name: str = HF_MODEL_ID,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 2048,
        dtype: str = "float32",
        attention_backend: Optional[str] = None,
    ):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self._model_name = model_name
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._attention_backend = attention_backend

        self._llm = None
        self._tokenizer = None
        self._decoder = None
        self._config = None
        self._data_processor = None
        self._collator = None
        self._labels: List[str] = []

    def _ensure_llm(self):
        if self._llm is not None:
            return
        from vllm import LLM

        import plugins.modernbert_gliner_rerank  # noqa: F401
        from plugins.deberta_gliner_linker.vllm_pooling_attention_mask import (
            apply_pooling_attention_mask_patch,
        )
        from plugins.modernbert_gliner_rerank import get_model_path

        apply_pooling_attention_mask_patch()

        llm_kw = dict(
            model=get_model_path(),
            trust_remote_code=True,
            dtype=self._dtype,
            max_model_len=self._max_model_len,
            enforce_eager=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=self._gpu_memory_utilization,
        )
        if self._attention_backend:
            llm_kw["attention_backend"] = self._attention_backend
        self._llm = LLM(**llm_kw)
        logger.info("vLLM LLM ready (modernbert_gliner_rerank)")

    def warmup(self, labels: List[str]) -> None:
        """Load tokenizer, decoder, collator from GLiNER and start vLLM (no label embedding precompute)."""
        unique = list(dict.fromkeys(labels))

        self._ensure_llm()

        from gliner import GLiNER
        from gliner.data_processing.collator import TokenDataCollator

        gliner = GLiNER.from_pretrained(self._model_name)
        if torch.cuda.is_available():
            gliner.to("cuda")

        self._labels = unique
        dp = gliner.data_processor
        self._tokenizer = dp.transformer_tokenizer
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

        logger.info(
            "warmup complete: %d labels, vLLM active (uni-encoder / L4 rerank path)", len(unique)
        )

    def _tokenize(self, text: str) -> dict:
        if self._data_processor is None or self._collator is None:
            raise RuntimeError("Call warmup(labels) first")

        words: List[str] = []
        word_starts: List[int] = []
        word_ends: List[int] = []
        for token, start, end in self._data_processor.words_splitter(text):
            words.append(token)
            word_starts.append(start)
            word_ends.append(end)

        batch = self._collator(
            [{"tokenized_text": words, "ner": None}],
            entity_types=self._labels,
        )

        input_ids = batch["input_ids"][0].detach().cpu()
        words_mask = batch["words_mask"][0].detach().cpu()
        tl = batch["text_lengths"]
        if tl.dim() == 2:
            text_length = int(tl[0, 0].item())
        else:
            text_length = int(tl[0].item())

        return {
            "text": text,
            "words": words,
            "word_starts": word_starts,
            "word_ends": word_ends,
            "words_mask": words_mask,
            "text_lengths": text_length,
            "prompt_token_ids": input_ids.tolist(),
        }

    def _run_vllm(self, tokenized: List[dict]) -> List[torch.Tensor]:
        from vllm.inputs import TokensPrompt
        from vllm.pooling_params import PoolingParams

        prompts, params = [], []
        for t in tokenized:
            ids = t["prompt_token_ids"]
            data = {
                "words_mask": t["words_mask"].tolist(),
                "text_lengths": t["text_lengths"],
            }
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(PoolingParams(task="plugin", extra_kwargs=data))

        outputs = self._llm.encode(
            prompts,
            pooling_params=params,
            pooling_task="plugin",
        )

        results = []
        for out in outputs:
            results.append(torch.tensor(out.outputs.data))
        return results

    def _decode(
        self,
        raw: torch.Tensor,
        tok: dict,
        *,
        threshold: float,
        flat_ner: bool,
        multi_label: bool,
    ) -> List[Dict[str, Any]]:
        if raw.numel() < 4:
            return []

        W = int(raw[0].item())
        C = int(raw[1].item())
        S = int(raw[2].item())
        scores = raw[3:].reshape(1, W, C, S)

        id_to_classes = {i + 1: label for i, label in enumerate(self._labels)}

        spans = self._decoder.decode(
            tokens=[tok["words"]],
            id_to_classes=id_to_classes,
            model_output=scores,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
        )

        src = tok.get("text")
        entities = []
        for span in spans[0]:
            ws = span.start
            we = span.end
            char_start = tok["word_starts"][ws] if ws < len(tok["word_starts"]) else 0
            char_end = tok["word_ends"][we] if we < len(tok["word_ends"]) else len(src or "")
            if src is not None:
                entity_text = src[char_start:char_end]
            else:
                entity_text = " ".join(tok["words"][ws : we + 1])

            entities.append(
                {
                    "start": char_start,
                    "end": char_end,
                    "text": entity_text,
                    "label": span.entity_type,
                    "score": round(span.score, 4),
                }
            )

        return entities

    def predict_entities(
        self,
        text: str,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> List[Dict[str, Any]]:
        return self.batch_predict_entities([text], threshold, flat_ner, multi_label)[0]

    def batch_predict_entities(
        self,
        texts: List[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        if not self._labels:
            raise RuntimeError("Call warmup(labels) first")

        tokenized = [self._tokenize(t) for t in texts]
        raw_outputs = self._run_vllm(tokenized)

        all_entities = []
        for raw, tok in zip(raw_outputs, tokenized):
            ents = self._decode(
                raw,
                tok,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )
            all_entities.append(ents)

        return all_entities

    def close(self):
        if self._llm is not None:
            del self._llm
            self._llm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("GLiNERRerankProcessor: closed")
