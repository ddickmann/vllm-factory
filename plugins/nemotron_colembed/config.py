"""NemotronColEmbed Config — extends Qwen3VLConfig with embedding pooling param."""
from transformers.models.qwen3_vl import Qwen3VLConfig


class NemotronColEmbedConfig(Qwen3VLConfig):
    model_type = "qwen3_vl_nemotron_embed"

    def __init__(self, pooling: str = "colbert", **kwargs):
        super().__init__(**kwargs)
        self.pooling = pooling
        # Enable bidirectional attention — this is checked by
        # Qwen3DecoderLayer (vLLM's qwen3.py:176) to set
        # AttentionType.ENCODER_ONLY instead of DECODER.
        self.is_causal = False
        if hasattr(self, "text_config") and self.text_config is not None:
            self.text_config.is_causal = False

