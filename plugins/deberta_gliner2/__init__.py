"""GLiNER2 Plugin — schema-based multi-task extraction with DeBERTa v2."""

from forge.registration import register_plugin

from .config import GLiNER2Config
from .model import GLiNER2VLLMModel


def register() -> None:
    register_plugin("gliner2", GLiNER2Config,
                     "GLiNER2VLLMModel", GLiNER2VLLMModel)


register()

__all__ = ["GLiNER2VLLMModel", "GLiNER2Config"]
