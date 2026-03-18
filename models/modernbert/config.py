"""
ModernColBERT Configuration.

This module defines the configuration class for ModernColBERT models,
extending ModernBertConfig with ColBERT-specific parameters.
"""


from transformers import ModernBertConfig


class ModernColBERTConfig(ModernBertConfig):
    """
    Configuration class for ModernColBERT model.

    Extends ModernBertConfig with ColBERT-specific parameters:
    - colbert_dim: Output embedding dimension (default: 128)
    - query_length: Maximum query length with padding (default: 256)
    - document_length: Maximum document length (default: 8192)

    Args:
        colbert_dim: Output embedding dimension for ColBERT
        query_length: Length to pad queries to for MaxSim operations
        document_length: Maximum length for documents
        **kwargs: Additional arguments passed to ModernBertConfig
    """

    model_type = "moderncolbert"

    def __init__(
        self,
        colbert_dim: int = 128,
        query_length: int = 256,
        document_length: int = 8192,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ColBERT specific parameters
        self.colbert_dim = colbert_dim
        self.query_length = query_length
        self.document_length = document_length

        # Auto mapping for transformers
        self.auto_map = {
            "AutoConfig": "config.ModernColBERTConfig",
            "AutoModel": "model.ModernBertForColBERT",
        }


def get_moderncolbert_config(model_name_or_path: str) -> ModernColBERTConfig:
    """
    Load ModernColBERT configuration from a model path or HuggingFace Hub.

    This function loads the model configuration and ensures it is converted
    to the ModernColBERTConfig format for compatibility with vLLM.

    Args:
        model_name_or_path: Path to local model directory or HuggingFace model ID

    Returns:
        ModernColBERTConfig: The model configuration

    Raises:
        ValueError: If the config cannot be loaded or converted
        OSError: If the model path does not exist
    """
    from vllm.transformers_utils.config import get_config

    try:
        config = get_config(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        raise ValueError(
            f"Failed to load config from '{model_name_or_path}': {str(e)}"
        ) from e

    # Convert to ModernColBERTConfig if needed
    if not isinstance(config, ModernColBERTConfig):
        try:
            config_dict = config.to_dict()
            config = ModernColBERTConfig(**config_dict)
        except Exception as e:
            raise ValueError(
                f"Failed to convert config to ModernColBERTConfig: {str(e)}"
            ) from e

    return config


