"""
LFM2ColBERT Configuration.

Extends Lfm2Config with ColBERT-specific parameters for
multi-vector retrieval with MaxSim scoring.
"""

from transformers import Lfm2Config


class LFM2ColBERTConfig(Lfm2Config):
    """Configuration for LFM2ColBERT model.

    Adds ColBERT-specific parameters to the standard LFM2 config:
    - colbert_dim: Output embedding dimension (default: 128)
    - query_length: Maximum query length with padding for MaxSim (default: 32)
    - document_length: Maximum document length (default: 8192)
    """

    model_type = "lfm2_colbert"

    def __init__(
        self,
        colbert_dim: int = 128,
        query_length: int = 32,
        document_length: int = 8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.colbert_dim = colbert_dim
        self.query_length = query_length
        self.document_length = document_length
        self.auto_map = {
            "AutoConfig": "config.LFM2ColBERTConfig",
            "AutoModel": "model.LFM2ForColBERT",
        }


def get_lfm2colbert_config(model_name_or_path: str) -> LFM2ColBERTConfig:
    """Load and convert a config to LFM2ColBERTConfig."""
    from vllm.transformers_utils.config import get_config

    config = get_config(model_name_or_path, trust_remote_code=True)
    if not isinstance(config, LFM2ColBERTConfig):
        config = LFM2ColBERTConfig(**config.to_dict())
    return config
