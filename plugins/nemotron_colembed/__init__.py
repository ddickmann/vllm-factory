"""NemotronColEmbed — Qwen3-VL with bidirectional attention + L2-normalized ColBERT embeddings."""

from forge.registration import register_plugin

from .config import NemotronColEmbedConfig
from .model import NemotronColEmbedModel


def register() -> None:
    register_plugin(
        "qwen3_vl_nemotron_embed",
        NemotronColEmbedConfig,
        "NemotronColEmbedModel",
        NemotronColEmbedModel,
        aliases=["Qwen3VLNemotronEmbedModel"],
    )


register()
__all__ = ["NemotronColEmbedModel", "NemotronColEmbedConfig"]
