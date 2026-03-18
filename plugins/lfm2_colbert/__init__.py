"""LFM2-ColBERT — LFM2 encoder + ColBERT multi-vector pooler."""

from forge.registration import register_plugin

from .config import LFM2ColBERTConfig
from .model import LFM2ForColBERT


def register() -> None:
    register_plugin("lfm2_colbert", LFM2ColBERTConfig,
                     "Lfm2Model", LFM2ForColBERT,
                     aliases=["LFM2ForColBERT"])


register()
__all__ = ["LFM2ForColBERT", "LFM2ColBERTConfig"]
