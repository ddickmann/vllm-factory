"""mmBERT-GLiNER2 — ModernBERT encoder + GLiNER2 multi-task extraction pooler."""

from forge.registration import register_plugin

from .config import GLiNER2ModernBertConfig
from .model import GLiNER2ModernBertModel


def register() -> None:
    register_plugin(
        "mmbert_gliner2", GLiNER2ModernBertConfig, "GLiNER2ModernBertModel", GLiNER2ModernBertModel
    )


register()

__all__ = ["GLiNER2ModernBertModel", "GLiNER2ModernBertConfig"]
