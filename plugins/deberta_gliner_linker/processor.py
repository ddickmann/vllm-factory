"""
GLiNER-Linker Processor — GLiNER-native preprocessing with vLLM inference.

Uses the same stack as GLiNER.inference / GLinker L3:
  - data_processor.words_splitter for text → words + char spans
  - BiEncoderTokenDataCollator + entity_types → input_ids, words_mask, text_lengths
    (includes [ENT, label, …, SEP] prefix before text words)
  - GLiNER.encode_labels for KB label embeddings
  - TokenDecoder for BIO decoding

Only the text-encoder → scorer forward runs on vLLM (GLiNER does not use the checkpoint LSTM on this path).

Lifecycle::

    proc = GLiNERLinkerProcessor()
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

HF_MODEL_ID = "knowledgator/gliner-linker-large-v1.0"
# GLinker `DAGExecutor.precompute_embeddings` passes this into `encode_labels` (see glinker/core/dag.py).
_ENCODE_LABELS_BATCH_SIZE = 32


def _cap_labels_tokenizer_max_length(gliner, max_length: int) -> None:
    """Mirror GLinker ``L3Component._setup``: the labels DeBERTa tokenizer often reports a huge
    ``model_max_length``, so ``encode_labels`` uses ``padding='max_length'`` without actually padding.
    GLinker caps the tokenizer to ``L3Config.max_length`` (512 in ``create_simple``); without that,
    ``batch_size=1`` encodes each label with implicit per-sequence length while precompute batches
    with fixed-width padding — different label embeddings and large score drift vs the reference.
    """
    dp = getattr(gliner, "data_processor", None)
    if dp is None or not hasattr(dp, "labels_tokenizer"):
        return
    tok = dp.labels_tokenizer
    if getattr(tok, "model_max_length", 0) > 100000:
        tok.model_max_length = max_length


class GLiNERLinkerProcessor:
    """Entity linking via GLiNER-compatible tensors and vLLM for the text encoder stack."""

    def __init__(
        self,
        model_name: str = HF_MODEL_ID,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 512,
        dtype: str = "float32",
    ):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self._model_name = model_name
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype

        self._llm = None
        self._tokenizer = None
        self._decoder = None
        self._config = None
        self._data_processor = None
        self._collator = None
        self._labels: List[str] = []
        self._label_embeddings: Optional[torch.Tensor] = None

    def _ensure_llm(self):
        if self._llm is not None:
            return
        import plugins.deberta_gliner_linker  # noqa: F401
        from plugins.deberta_gliner_linker import get_model_path
        from plugins.deberta_gliner_linker.vllm_pooling_attention_mask import (
            apply_pooling_attention_mask_patch,
        )
        from vllm import LLM

        apply_pooling_attention_mask_patch()

        self._llm = LLM(
            model=get_model_path(),
            trust_remote_code=True,
            dtype=self._dtype,
            max_model_len=self._max_model_len,
            enforce_eager=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=self._gpu_memory_utilization,
        )
        logger.info("vLLM LLM ready")

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self, labels: List[str]) -> None:
        """Precompute label embeddings and initialize vLLM.

        Args:
            labels: Formatted label strings for the KB.
        """
        unique = list(dict.fromkeys(labels))

        # 1. Init vLLM first (clean fork, no heavy model in parent memory)
        self._ensure_llm()

        # 2. Load GLiNER, encode labels, extract components, then free it
        from gliner import GLiNER
        from gliner.data_processing.collator import BiEncoderTokenDataCollator

        gliner = GLiNER.from_pretrained(self._model_name)
        # Stay on CPU — vLLM already owns the GPU; label encoding runs once.

        _cap_labels_tokenizer_max_length(gliner, self._max_model_len)

        bs = min(_ENCODE_LABELS_BATCH_SIZE, len(unique))
        logger.info("Encoding %d labels (batch_size=%d)...", len(unique), bs)
        self._label_embeddings = gliner.encode_labels(unique, batch_size=bs).cpu()
        self._label_embeddings_list = self._label_embeddings.tolist()
        self._labels = unique

        dp = gliner.data_processor
        self._tokenizer = dp.transformer_tokenizer
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

        # Free the heavy model (keep data_processor + collator: no reference to gliner.model)
        del gliner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("warmup complete: %d labels, vLLM active", len(unique))

    # ------------------------------------------------------------------
    # Tokenization (matches GLiNER.inference collation)
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> dict:
        """Build input_ids / words_mask / text_lengths exactly like GLiNER inference."""
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

        return {
            "text": text,
            "words": words,
            "word_starts": word_starts,
            "word_ends": word_ends,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "words_mask": words_mask,
            "text_lengths": text_length,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_vllm(self, tokenized: List[dict]) -> List[torch.Tensor]:
        """Call vLLM LLM.embed() for a batch of tokenized texts."""
        from vllm.inputs import TokensPrompt
        from vllm.pooling_params import PoolingParams

        le_list = self._label_embeddings_list
        prompts, params = [], []

        for t in tokenized:
            ids = t["input_ids"].tolist()
            data = {
                "input_ids": ids,
                "attention_mask": t["attention_mask"].tolist(),
                "words_mask": t["words_mask"].tolist(),
                "text_lengths": t["text_lengths"],
                "labels_embeds": le_list,
            }
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(PoolingParams(extra_kwargs=data))

        outputs = self._llm.embed(prompts, pooling_params=params)

        results = []
        for out in outputs:
            results.append(torch.tensor(out.outputs.embedding))
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
        """Decode vLLM output using GLiNER's TokenDecoder (same API as GLiNER.inference)."""
        if raw.numel() < 4:
            return []

        W = int(raw[0].item())
        C = int(raw[1].item())
        S = int(raw[2].item())
        scores = raw[3:].reshape(1, W, C, S)  # (1, W, C, 3) — last dim start / end / inside

        id_to_classes = {i + 1: label for i, label in enumerate(self._labels)}

        # Must use the same threshold as GLiNER.decode: start/end candidates use
        # sigmoid(logits) > threshold; using 0.0 makes (sigmoid > 0) always true and
        # changes greedy_search / flat_ner vs the reference pipeline.
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
            # TokenDecoder uses inclusive end word index (inside scores are st : ed + 1).
            ws = span.start
            we = span.end
            char_start = tok["word_starts"][ws] if ws < len(tok["word_starts"]) else 0
            char_end = tok["word_ends"][we] if we < len(tok["word_ends"]) else len(src or "")
            # Match GLiNER entity formatting: substring of the original text (exclusive end).
            if src is not None:
                entity_text = src[char_start:char_end]
            else:
                entity_text = " ".join(tok["words"][ws : we + 1])

            entities.append({
                "start": char_start,
                "end": char_end,
                "text": entity_text,
                "label": span.entity_type,
                "score": round(span.score, 4),
            })

        return entities

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_entities(
        self,
        text: str,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> List[Dict[str, Any]]:
        """Predict entities in a single text."""
        return self.batch_predict_entities([text], threshold, flat_ner, multi_label)[0]

    def batch_predict_entities(
        self,
        texts: List[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """Predict entities in a batch of texts."""
        if not self._labels or self._label_embeddings is None:
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

    def encode_labels(self, labels: List[str]) -> torch.Tensor:
        """Encode labels using GLiNER's labels encoder."""
        from gliner import GLiNER

        gliner = GLiNER.from_pretrained(self._model_name)
        _cap_labels_tokenizer_max_length(gliner, self._max_model_len)
        bs = min(_ENCODE_LABELS_BATCH_SIZE, max(1, len(labels)))
        embs = gliner.encode_labels(labels, batch_size=bs).cpu()
        del gliner
        gc.collect()
        return embs

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        if self._llm is not None:
            del self._llm
            self._llm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("GLiNERLinkerProcessor: closed")
