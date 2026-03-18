"""DeBERTa v2 GLiNER Plugin — zero-shot NER with DeBERTa v2 backbone."""

from forge.registration import register_plugin

from .config import GLiNERDebertaV2Config
from .model import GLiNERDebertaV2Model


def register() -> None:
    register_plugin("gliner_deberta_v2", GLiNERDebertaV2Config,
                     "GLiNERDebertaV2Model", GLiNERDebertaV2Model)


register()

__all__ = ["GLiNERDebertaV2Model", "GLiNERDebertaV2Config"]
