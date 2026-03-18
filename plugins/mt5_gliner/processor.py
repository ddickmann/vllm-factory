"""
GLiNER mT5 Processor — async inference pipeline.

Uses shared forge/gliner_preprocessor.py and forge/gliner_postprocessor.py
(mirrored from superpod) for production-grade preprocessing and decoding.

Preprocessing:  Tokenize text + entity labels, build words_mask + span metadata
Engine:         AsyncLLMEngine.encode() with task="embed" + extra_kwargs
Postprocessing: Decode span logits → list of {text, label, score, start, end} dicts

MT5-specific: includes attention_mask in extra_kwargs (required by T5 encoder).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer
from vllm.inputs import TokensPrompt

from forge.gliner_postprocessor import GLiNERDecoder, get_final_entities
from forge.gliner_preprocessor import GLiNERPreprocessor
from forge.processor_base import BaseProcessor, PreprocessedInput

try:
    from vllm import PoolingParams
except ImportError:
    PoolingParams = None


class GLiNERMT5Processor(BaseProcessor):
    """Async processor for GLiNER span extraction with mT5 backbone.

    Usage:
        processor = GLiNERMT5Processor("my-mt5-gliner-model")
        results = await processor.process_batch(
            ["Le PDG d'Apple a visité Paris."],
            labels=["personne", "organisation", "lieu"],
        )
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True,
        )
        # Load config for GLiNERPreprocessor
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        self._preprocessor = GLiNERPreprocessor(
            underlying_tokenizer=self._tokenizer,
            config=config,
            device="cpu",
            include_attention_mask=True,  # MT5 requires attention_mask
        )
        self._decoder = GLiNERDecoder()

    def preprocess(self, text: str, **kwargs) -> PreprocessedInput:
        labels: List[str] = kwargs.get("labels", [])
        threshold: float = kwargs.get("threshold", 0.5)
        flat_ner: bool = kwargs.get("flat_ner", False)
        multi_label: bool = kwargs.get("multi_label", False)

        # Use shared preprocessor (single-item batch)
        result = self._preprocessor(text, labels, device="cpu")
        enc = result["model_inputs"]
        meta = result["postprocessing_metadata"]

        # Extract single-item from batch
        input_ids = enc["input_ids"][0]
        words_mask = enc["words_mask"][0]
        text_lengths = enc["text_lengths"][0].item()

        # Build extra_kwargs (matches superpod's model_requests.py)
        gliner_data = {
            "input_ids": input_ids.tolist(),
            "words_mask": words_mask.tolist(),
            "text_lengths": text_lengths,
            "attention_mask": enc["attention_mask"][0].tolist(),
        }

        prompt = TokensPrompt(prompt_token_ids=input_ids.tolist())
        pooling_params = PoolingParams(extra_kwargs=gliner_data)

        return PreprocessedInput(
            prompt=prompt,
            pooling_params=pooling_params,
            metadata={
                "text": text,
                "labels": labels,
                "threshold": threshold,
                "flat_ner": flat_ner,
                "multi_label": multi_label,
                "tokens": meta["tokens"],
                "word_positions": meta["word_positions"],
                "id_to_classes": meta["id_to_classes"],
            },
        )

    def postprocess(self, raw_output: Any, metadata: Optional[Dict] = None) -> List[Dict]:
        if raw_output is None or metadata is None:
            return []

        scores = torch.as_tensor(raw_output) if not isinstance(raw_output, torch.Tensor) else raw_output

        # Extract shape prefix [L, K, C] and reshape to (B=1, L, K, C)
        if scores.dim() == 1 and scores.numel() > 3:
            L = int(scores[0].item())
            K = int(scores[1].item())
            C = int(scores[2].item())
            logits = scores[3:].reshape(1, L, K, C)
        elif scores.dim() == 3:
            logits = scores.unsqueeze(0)
        else:
            return []

        # Decode using shared GLiNERDecoder
        decoded = self._decoder.decode(
            tokens=metadata["tokens"],
            id_to_classes=metadata["id_to_classes"],
            logits=logits,
            flat_ner=metadata.get("flat_ner", False),
            threshold=metadata.get("threshold", 0.5),
            multi_label=metadata.get("multi_label", False),
        )

        # Convert to character-offset entities
        entities_batch = get_final_entities(
            decoded_outputs=decoded,
            word_positions=metadata["word_positions"],
            original_texts=[metadata["text"]],
        )

        return entities_batch[0] if entities_batch else []
