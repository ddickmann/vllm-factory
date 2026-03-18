"""mmBERT-GLiNER — ModernBERT encoder + GLiNER span extraction pooler."""

from forge.registration import register_plugin

from .config import GLiNERModernBertConfig
from .model import GLiNERModernBertModel


def register() -> None:
    register_plugin("gliner_mmbert", GLiNERModernBertConfig,
                     "GLiNERModernBertModel", GLiNERModernBertModel)


register()

__all__ = ["GLiNERModernBertModel", "GLiNERModernBertConfig"]
