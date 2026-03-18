"""
ColLFM2 Configuration

Extends Lfm2VlConfig to add ColPali-specific parameters.
The model_type is set to "collfm2" for VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1.

IMPORTANT: The ColLFM2 model config from HuggingFace is minimal and only contains:
- architectures, model_type, base_model, dim, hidden_size, mask_non_image_embeddings

vLLM's Lfm2VLForConditionalGeneration expects full nested configs (text_config, vision_config).
This class automatically fetches and merges the full config from the base model when needed.
"""

from transformers.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig

# Cache for base config to avoid repeated downloads
_BASE_CONFIG_CACHE = {}


class ColLFM2Config(Lfm2VlConfig):
    """Configuration for ColLFM2 (ColPali with LFM2-VL backbone).

    This config extends Lfm2VlConfig and adds:
    - dim: ColPali embedding dimension (default 128)
    - mask_non_image_embeddings: Whether to mask non-image tokens
    - base_model: Reference to base LFM2-VL model for config inheritance

    When initialized with a minimal config (missing text_config/vision_config),
    automatically fetches the full config from the base model and merges it.
    """
    model_type = "collfm2"

    # Fields that must be inherited from base model if missing
    _BASE_CONFIG_FIELDS = [
        "text_config",
        "vision_config",
        "projector_hidden_size",
        "projector_bias",
        "projector_hidden_act",
        "downsample_factor",
        "image_token_id",
        "image_token_index",
        "max_image_tokens",
        "min_image_tokens",
        "max_num_patches",
        "max_tiles",
        "min_tiles",
        "tile_size",
        "encoder_patch_size",
        "do_image_splitting",
        "use_thumbnail",
        "use_image_special_tokens",
        "max_pixels_tolerance",
    ]

    def __init__(
        self,
        dim: int = 128,
        mask_non_image_embeddings: bool = False,
        base_model: str = "LiquidAI/LFM2-VL-450M",
        **kwargs,
    ):
        # Check if we need to fetch base config (missing critical nested configs)
        needs_base_config = (
            "text_config" not in kwargs or kwargs.get("text_config") is None or
            "vision_config" not in kwargs or kwargs.get("vision_config") is None
        )

        if needs_base_config:
            # Fetch and merge base config
            base_config = self._fetch_base_config(base_model)
            kwargs = self._merge_base_config(base_config, kwargs)

        # Call parent init with merged kwargs
        super().__init__(**kwargs)

        # Set ColLFM2-specific attributes
        self.dim = dim
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.base_model = base_model

        # Ensure vLLM-specific defaults are set as attributes
        # These might not be set by parent __init__ if not in kwargs
        for field, default_value in self._VLLM_DEFAULTS.items():
            if not hasattr(self, field) or getattr(self, field) is None:
                setattr(self, field, default_value)

    @classmethod
    def _fetch_base_config(cls, base_model: str):
        """Fetch configuration from base model (cached)."""
        if base_model not in _BASE_CONFIG_CACHE:
            from transformers import AutoConfig
            print(f"[ColLFM2Config] Fetching base config from {base_model}...")
            _BASE_CONFIG_CACHE[base_model] = AutoConfig.from_pretrained(
                base_model,
                trust_remote_code=True
            )
        return _BASE_CONFIG_CACHE[base_model]

    # vLLM-specific fields that may not be in HuggingFace config but are required by vLLM
    # LFM2-VL-450M checkpoint has layer_norm in multi_modal_projector, so default to True
    _VLLM_DEFAULTS = {
        "projector_use_layernorm": True,  # vLLM expects this but HF doesn't have it
    }

    @classmethod
    def _merge_base_config(cls, base_config, kwargs: dict) -> dict:
        """Merge base config fields into kwargs."""
        merged = kwargs.copy()

        for field in cls._BASE_CONFIG_FIELDS:
            # Only add if not already in kwargs
            if field not in merged or merged.get(field) is None:
                if hasattr(base_config, field):
                    value = getattr(base_config, field)
                    # For nested configs (text_config, vision_config), convert to dict
                    if hasattr(value, 'to_dict'):
                        merged[field] = value.to_dict()
                    else:
                        merged[field] = value

        # Add vLLM-specific defaults that might not be in HF config
        for field, default_value in cls._VLLM_DEFAULTS.items():
            if field not in merged or merged.get(field) is None:
                if not hasattr(base_config, field):
                    merged[field] = default_value

        return merged
