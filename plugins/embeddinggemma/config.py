"""EmbeddingGemma Config — extends Gemma3TextConfig for embedding tasks."""

from transformers import Gemma3TextConfig


class EmbeddingGemmaConfig(Gemma3TextConfig):
    """Config for EmbeddingGemma (gemma3_text backbone + 2 Dense projection layers).

    Adds:
        embedding_dim:  Final embedding dimension (default 768, Matryoshka truncatable)
        dense1_out:     First dense layer output dim (default 3072)
        is_causal:      Set False for bidirectional (encoder-only) attention
    """

    model_type = "embedding_gemma"

    def __init__(
        self,
        embedding_dim: int = 768,
        dense1_out: int = 3072,
        **kwargs,
    ):
        # Force encoder-only (bidirectional) attention
        kwargs.setdefault("is_causal", False)
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.dense1_out = dense1_out
        # Ensure bidirectional attention is enabled
        self.is_causal = False
