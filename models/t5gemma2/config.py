"""T5Gemma2 config helpers for vLLM Factory."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

try:
    from transformers import SiglipVisionConfig
except ImportError:
    SiglipVisionConfig = PretrainedConfig

try:
    from transformers import (
        T5Gemma2Config as HFT5Gemma2Config,
        T5Gemma2DecoderConfig as HFT5Gemma2DecoderConfig,
        T5Gemma2EncoderConfig as HFT5Gemma2EncoderConfig,
        T5Gemma2TextConfig as HFT5Gemma2TextConfig,
    )
    HAS_NATIVE_T5GEMMA2_CONFIG = True
except ImportError:
    HFT5Gemma2Config = PretrainedConfig
    HFT5Gemma2TextConfig = PretrainedConfig
    HFT5Gemma2EncoderConfig = PretrainedConfig
    HFT5Gemma2DecoderConfig = PretrainedConfig
    HAS_NATIVE_T5GEMMA2_CONFIG = False


if HAS_NATIVE_T5GEMMA2_CONFIG:

    class T5Gemma2TextConfig(HFT5Gemma2TextConfig):
        pass


    class T5Gemma2EncoderConfig(HFT5Gemma2EncoderConfig):
        pass


    class T5Gemma2DecoderConfig(HFT5Gemma2DecoderConfig):
        pass


    class T5Gemma2Config(HFT5Gemma2Config):
        """Thin wrapper over the upstream HF T5Gemma2 config."""

        model_type = "t5gemma2"

else:

    class T5Gemma2TextConfig(PretrainedConfig):
        """Fallback copy of the upstream HF T5Gemma2 text config."""

        model_type = "t5gemma2_text"
        default_theta = {"global": 1_000_000.0, "local": 10_000.0}

        def __init__(
            self,
            vocab_size: int = 262_208,
            hidden_size: int = 2304,
            intermediate_size: int = 9216,
            num_hidden_layers: int = 26,
            num_attention_heads: int = 8,
            num_key_value_heads: int = 4,
            head_dim: int = 256,
            hidden_activation: str = "gelu_pytorch_tanh",
            max_position_embeddings: int = 131_072,
            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-6,
            use_cache: bool = True,
            pad_token_id: int | None = 0,
            eos_token_id: int | list[int] | None = 1,
            bos_token_id: int | None = 2,
            tie_word_embeddings: bool = True,
            rope_parameters: dict[str, Any] | None = None,
            attention_bias: bool = False,
            attention_dropout: float = 0.0,
            query_pre_attn_scalar: int = 256,
            sliding_window: int | None = 4096,
            layer_types: list[str] | None = None,
            final_logit_softcapping: float | None = None,
            attn_logit_softcapping: float | None = None,
            dropout_rate: float = 0.0,
            sliding_window_pattern: int = 6,
            **kwargs: Any,
        ) -> None:
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.head_dim = head_dim
            self.hidden_activation = hidden_activation
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.rms_norm_eps = rms_norm_eps
            self.use_cache = use_cache
            self.tie_word_embeddings = tie_word_embeddings
            self.rope_parameters = rope_parameters
            self.attention_bias = attention_bias
            self.attention_dropout = attention_dropout
            self.query_pre_attn_scalar = query_pre_attn_scalar
            self.sliding_window = sliding_window
            self.layer_types = layer_types
            self.final_logit_softcapping = final_logit_softcapping
            self.attn_logit_softcapping = attn_logit_softcapping
            self.dropout_rate = dropout_rate

            if self.layer_types is None:
                self.layer_types = [
                    "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                    for i in range(self.num_hidden_layers)
                ]

            default_rope_params = {
                "sliding_attention": {"rope_type": "default"},
                "full_attention": {"rope_type": "default"},
            }
            self.rope_parameters = self.rope_parameters or default_rope_params
            self.rope_parameters.setdefault("full_attention", {"rope_type": "default"})
            self.rope_parameters.setdefault("sliding_attention", {"rope_type": "default"})
            self.rope_parameters["full_attention"].setdefault(
                "rope_theta",
                self.default_theta["global"],
            )
            self.rope_parameters["sliding_attention"].setdefault(
                "rope_theta",
                self.default_theta["local"],
            )

            super().__init__(
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )


    class T5Gemma2EncoderConfig(PretrainedConfig):
        """Fallback copy of the upstream HF T5Gemma2 encoder config."""

        model_type = "t5gemma2_encoder"

        def __init__(
            self,
            text_config: T5Gemma2TextConfig | dict[str, Any] | None = None,
            vision_config: SiglipVisionConfig | dict[str, Any] | None = None,
            mm_tokens_per_image: int = 256,
            boi_token_index: int = 255_999,
            eoi_token_index: int = 256_000,
            image_token_index: int = 262_144,
            initializer_range: float = 0.02,
            tie_word_embeddings: bool = True,
            **kwargs: Any,
        ) -> None:
            if text_config is None:
                text_config = T5Gemma2TextConfig()
            elif isinstance(text_config, dict):
                text_config = T5Gemma2TextConfig(**text_config)

            if vision_config is None:
                vision_config = SiglipVisionConfig() if SiglipVisionConfig is not PretrainedConfig else PretrainedConfig()
            elif isinstance(vision_config, dict):
                vision_config = (
                    SiglipVisionConfig(**vision_config)
                    if SiglipVisionConfig is not PretrainedConfig
                    else PretrainedConfig(**vision_config)
                )

            self.text_config = text_config
            self.vision_config = vision_config
            self.mm_tokens_per_image = mm_tokens_per_image
            self.boi_token_index = boi_token_index
            self.eoi_token_index = eoi_token_index
            self.image_token_index = image_token_index
            self.initializer_range = initializer_range
            self.tie_word_embeddings = tie_word_embeddings
            super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


    class T5Gemma2DecoderConfig(T5Gemma2TextConfig):
        """Fallback copy of the upstream HF T5Gemma2 decoder config."""

        model_type = "t5gemma2_decoder"


    class T5Gemma2Config(PretrainedConfig):
        """Fallback top-level config matching the upstream HF layout."""

        model_type = "t5gemma2"

        def __init__(
            self,
            encoder: T5Gemma2EncoderConfig | dict[str, Any] | None = None,
            decoder: T5Gemma2DecoderConfig | dict[str, Any] | None = None,
            is_encoder_decoder: bool = True,
            dropout_rate: float = 0.0,
            attention_dropout: float = 0.0,
            classifier_dropout_rate: float = 0.0,
            initializer_range: float = 0.02,
            image_token_index: int = 256_001,
            eoi_token_index: int | None = None,
            tie_word_embeddings: bool = True,
            **kwargs: Any,
        ) -> None:
            if encoder is None:
                encoder = T5Gemma2EncoderConfig()
            elif isinstance(encoder, dict):
                encoder = T5Gemma2EncoderConfig(**encoder)

            if decoder is None:
                decoder = T5Gemma2DecoderConfig()
            elif isinstance(decoder, dict):
                decoder = T5Gemma2DecoderConfig(**decoder)

            encoder.text_config.dropout_rate = dropout_rate
            encoder.text_config.attention_dropout = attention_dropout
            decoder.dropout_rate = dropout_rate
            decoder.attention_dropout = attention_dropout
            encoder.image_token_index = image_token_index

            self.encoder = encoder
            self.decoder = decoder
            self.is_encoder_decoder = is_encoder_decoder
            self.dropout_rate = dropout_rate
            self.attention_dropout = attention_dropout
            self.classifier_dropout_rate = classifier_dropout_rate
            self.initializer_range = initializer_range
            self.image_token_index = image_token_index
            self.eoi_token_index = encoder.eoi_token_index if eoi_token_index is None else eoi_token_index
            self.tie_word_embeddings = tie_word_embeddings

            super().__init__(
                pad_token_id=decoder.pad_token_id,
                bos_token_id=decoder.bos_token_id,
                eos_token_id=decoder.eos_token_id,
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )


def require_t5gemma2_config() -> None:
    """Raise a clear error when the installed transformers is too old."""

    if not HAS_NATIVE_T5GEMMA2_CONFIG:
        return


def get_t5gemma2_text_config(
    config: T5Gemma2Config,
    *,
    is_encoder: bool = True,
) -> T5Gemma2TextConfig | T5Gemma2DecoderConfig:
    """Return the text config for the encoder or decoder branch."""

    require_t5gemma2_config()
    return config.encoder.text_config if is_encoder else config.decoder


def get_t5gemma2_text_config_dict(
    config: T5Gemma2Config,
    *,
    is_encoder: bool = True,
) -> dict[str, Any]:
    """Flatten the relevant text config into a simple dict."""

    text_config = get_t5gemma2_text_config(config, is_encoder=is_encoder)
    return {
        "vocab_size": text_config.vocab_size,
        "hidden_size": text_config.hidden_size,
        "intermediate_size": text_config.intermediate_size,
        "num_hidden_layers": text_config.num_hidden_layers,
        "num_attention_heads": text_config.num_attention_heads,
        "num_key_value_heads": text_config.num_key_value_heads,
        "head_dim": text_config.head_dim,
        "max_position_embeddings": text_config.max_position_embeddings,
        "rms_norm_eps": text_config.rms_norm_eps,
        "dropout_rate": getattr(text_config, "dropout_rate", 0.0),
        "attention_dropout": getattr(text_config, "attention_dropout", 0.0),
        "query_pre_attn_scalar": getattr(
            text_config,
            "query_pre_attn_scalar",
            text_config.head_dim,
        ),
        "layer_types": list(text_config.layer_types),
        "sliding_window": text_config.sliding_window,
        "rope_parameters": text_config.rope_parameters,
        "hidden_act": text_config.hidden_activation,
        "attn_logit_softcapping": text_config.attn_logit_softcapping,
        "final_logit_softcapping": text_config.final_logit_softcapping,
        "attention_bias": getattr(text_config, "attention_bias", False),
        "pad_token_id": text_config.pad_token_id,
        "bos_token_id": getattr(text_config, "bos_token_id", None),
        "eos_token_id": getattr(text_config, "eos_token_id", None),
        "tie_word_embeddings": getattr(text_config, "tie_word_embeddings", True),
    }


__all__ = [
    "T5Gemma2Config",
    "T5Gemma2TextConfig",
    "T5Gemma2EncoderConfig",
    "T5Gemma2DecoderConfig",
    "get_t5gemma2_text_config",
    "get_t5gemma2_text_config_dict",
    "require_t5gemma2_config",
]
