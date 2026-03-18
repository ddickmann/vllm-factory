"""
ModernColBERT Configuration.

Extends ModernBertConfig with ColBERT-specific parameters for
multi-vector retrieval with MaxSim scoring.
"""


from transformers import ModernBertConfig


class ModernColBERTConfig(ModernBertConfig):
    """Configuration for ModernColBERT model.

    Adds ColBERT-specific parameters to the standard ModernBERT config:
    - colbert_dim: Output embedding dimension (default: 128)
    - query_length: Maximum query length with padding for MaxSim (default: 256)
    - document_length: Maximum document length (default: 8192)
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
        self.colbert_dim = colbert_dim
        self.query_length = query_length
        self.document_length = document_length
        self.auto_map = {
            "AutoConfig": "config.ModernColBERTConfig",
            "AutoModel": "model.ModernBertForColBERT",
        }


def get_moderncolbert_config(model_name_or_path: str) -> ModernColBERTConfig:
    """Load and convert a config to ModernColBERTConfig."""
    from vllm.transformers_utils.config import get_config

    config = get_config(model_name_or_path, trust_remote_code=True)
    if not isinstance(config, ModernColBERTConfig):
        config = ModernColBERTConfig(**config.to_dict())
    return config
