"""
GLiNER Preprocessor — shared preprocessing for GLiNER span extraction.

Mirrors superpod's gliner_preprocessor.py exactly.
Used by both mmbert_gliner and mt5_gliner plugins.

Key features:
1. Regex-based word splitting (matches GLiNER library's behavior)
2. Vectorized word mask generation via tokenizer.word_ids()
3. Vectorized span index and mask generation
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer


class GLiNERPreprocessor:
    """
    Production-grade preprocessor for GLiNER models.

    Supports batched preprocessing with vectorized operations for
    word masks, span indices, and span masks.
    """
    def __init__(
        self,
        underlying_tokenizer: PreTrainedTokenizer,
        config: Any,
        device: Optional[str] = "cpu",
        include_attention_mask: bool = False
    ):
        self.tokenizer = underlying_tokenizer
        self.max_len = getattr(config, 'max_len', 1024)
        self.max_width = getattr(config, 'max_width', 12)
        self.ent_token = getattr(config, 'ent_token', "<<ENT>>")
        self.sep_token = getattr(config, 'sep_token', "<<SEP>>")
        self.device = device
        self.include_attention_mask = include_attention_mask

        # Exact original regex pattern from GLiNER
        self.word_pattern = re.compile(r'\w+(?:[-_]\w+)*|\S')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'

    def _split_text_into_words(self, text: str) -> List[Tuple[str, int, int]]:
        """Splits text into words with their start and end character positions."""
        return [
            (match.group(), match.start(), match.end())
            for match in self.word_pattern.finditer(text)
        ]

    @torch.no_grad()
    def _create_span_tensors_vectorized(
        self, seq_lengths: List[int], max_seq_len: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized generation of span indices and masks."""
        B = len(seq_lengths)
        K = self.max_width

        starts = torch.arange(max_seq_len, device=device).unsqueeze(1)
        widths = torch.arange(K, device=device).unsqueeze(0)

        span_starts = starts.expand(-1, K)
        span_ends = span_starts + widths

        span_indices_proto = torch.stack([span_starts, span_ends], dim=-1).view(-1, 2)
        span_indices_batch = span_indices_proto.unsqueeze(0).expand(B, -1, -1)

        seq_lengths_tensor = torch.as_tensor(seq_lengths, device=device).view(B, 1, 1)
        span_starts_b = span_starts.unsqueeze(0)
        span_ends_b = span_ends.unsqueeze(0)

        span_mask = (span_starts_b < seq_lengths_tensor) & \
                    (span_ends_b < seq_lengths_tensor)

        span_masks_batch = span_mask.view(B, -1)

        return span_indices_batch, span_masks_batch

    @torch.no_grad()
    def _create_word_masks_vectorized(
        self,
        tokenized_inputs: Any,
        prompt_lengths: List[int],
        device: str
    ) -> torch.Tensor:
        """Vectorized generation of word-level masks from tokenizer.word_ids()."""
        word_masks = []
        for i, prompt_len in enumerate(prompt_lengths):
            word_ids_list = tokenized_inputs.word_ids(batch_index=i)

            word_ids = torch.tensor(
                [w if w is not None else -1 for w in word_ids_list],
                device=device,
                dtype=torch.long
            )

            prev_word_ids = torch.roll(word_ids, 1, dims=0)
            prev_word_ids[0] = -1

            is_new_word = (word_ids != -1) & (word_ids != prev_word_ids)
            is_in_text = (word_ids >= prompt_len)
            valid_indices = is_new_word & is_in_text

            word_mask = torch.zeros_like(word_ids)
            word_mask[valid_indices] = word_ids[valid_indices] - prompt_len + 1

            word_masks.append(word_mask)

        return torch.stack(word_masks, dim=0)

    def __call__(self, texts: Union[str, List[str]], labels: List[str], device: Optional[str] = None) -> Dict[str, Any]:
        """Preprocess raw text into model inputs and postprocessing metadata."""
        if not labels:
            raise ValueError("The `labels` list cannot be empty.")

        target_device = device if device is not None else self.device
        texts_to_process = [texts] if isinstance(texts, str) else texts
        B = len(texts_to_process)

        # 1. Prepare word-level data and prompts
        prompt_list = [token for entity in labels for token in (self.ent_token, entity)]
        prompt_list.append(self.sep_token)
        prompt_len = len(prompt_list)

        all_words, all_word_positions, seq_lengths = [], [], []
        input_prompts = []

        for text in texts_to_process:
            words_with_pos = self._split_text_into_words(text)
            words = [w[0] for w in words_with_pos][:self.max_len]
            positions = [(w[1], w[2]) for w in words_with_pos][:self.max_len]

            all_words.append(words)
            all_word_positions.append(positions)
            seq_lengths.append(len(words))
            input_prompts.append(prompt_list + words)

        prompt_lengths = [prompt_len] * B

        # 2. Tokenize using is_split_into_words=True
        tokenized_inputs = self.tokenizer(
            input_prompts,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            padding='longest'
        )

        # 3. Create word-level masks
        words_mask_tensor = self._create_word_masks_vectorized(
            tokenized_inputs, prompt_lengths, target_device
        )

        # 4. Generate span indices and masks
        max_seq_len = max(seq_lengths) if seq_lengths else 0
        span_indices_tensor, span_masks_tensor = self._create_span_tensors_vectorized(
            seq_lengths, max_seq_len, target_device
        )

        # 5. Assemble model inputs
        model_inputs = {
            'input_ids': tokenized_inputs['input_ids'].to(target_device),
            'words_mask': words_mask_tensor,
            'span_idx': span_indices_tensor,
            'span_mask': span_masks_tensor,
            'text_lengths': torch.as_tensor(seq_lengths, dtype=torch.long, device=target_device).unsqueeze(-1),
        }

        if self.include_attention_mask:
            model_inputs['attention_mask'] = tokenized_inputs['attention_mask'].to(target_device)

        postprocessing_metadata = {
            "tokens": all_words,
            "word_positions": all_word_positions,
            "seq_lengths": seq_lengths,
            "id_to_classes": {i + 1: label for i, label in enumerate(labels)},
            "entities": [[] for _ in texts_to_process],
        }

        return {
            "model_inputs": model_inputs,
            "postprocessing_metadata": postprocessing_metadata
        }
