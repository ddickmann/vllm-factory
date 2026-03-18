"""DeBERTa v1 configuration for vLLM Factory."""

from transformers import DebertaConfig


class DebertaVllmConfig(DebertaConfig):
    """DeBERTa v1 config extending HuggingFace DebertaConfig for vLLM."""
    model_type = "deberta_vllm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Auto mapping for transformers
        self.auto_map = {
            "AutoConfig": "config.DebertaVllmConfig",
            "AutoModel": "deberta_encoder.DebertaEncoderModel",
        }


__all__ = ["DebertaVllmConfig"]
