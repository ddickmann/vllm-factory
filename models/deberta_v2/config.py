"""DeBERTa v2/v3 configuration for vLLM Factory."""

from transformers import DebertaV2Config


class DebertaV2VllmConfig(DebertaV2Config):
    """DeBERTa v2/v3 config extending HuggingFace DebertaV2Config for vLLM."""
    model_type = "deberta_v2_vllm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Auto mapping for transformers
        self.auto_map = {
            "AutoConfig": "config.DebertaV2VllmConfig",
            "AutoModel": "deberta_v2_encoder.DebertaV2EncoderModel",
        }


__all__ = ["DebertaV2VllmConfig"]
