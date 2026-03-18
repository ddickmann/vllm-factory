"""EmbeddingGemma vLLM Plugin — Gemma with MEAN + Dense projection + L2 norm.

Pooling is handled by vLLM's built-in DispatchPooler which auto-loads
the SentenceTransformers Dense layers from the HF repo.
"""

from forge.registration import register_plugin

from .config import EmbeddingGemmaConfig
from .model import EmbeddingGemmaModel


def register() -> None:
    register_plugin("embedding_gemma", EmbeddingGemmaConfig,
                     "EmbeddingGemmaModel", EmbeddingGemmaModel)


register()

__all__ = ["EmbeddingGemmaModel", "EmbeddingGemmaConfig"]
