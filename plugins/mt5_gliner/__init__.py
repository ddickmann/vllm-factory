"""MT5 GLiNER Plugin — multilingual zero-shot NER with mT5 backbone."""

from forge.registration import register_plugin

from .config import GLiNERMT5Config
from .model import GLiNERMT5Model


def register() -> None:
    register_plugin("gliner_mt5", GLiNERMT5Config,
                     "GLiNERMT5Model", GLiNERMT5Model)


register()

__all__ = ["GLiNERMT5Model", "GLiNERMT5Config"]
