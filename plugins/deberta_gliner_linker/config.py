"""GLiNER-Linker Configuration — DeBERTa v1 bi-encoder for entity embedding.

Uses PretrainedConfig with num_hidden_layers=0 so vLLM skips KV cache
allocation. The actual DeBERTa v1 encoder config is stored as separate
attributes and used by the model to construct HF DebertaModel.

Reads from gliner_config.json → labels_encoder_config section.
"""

from transformers import PretrainedConfig


class GLiNERLinkerConfig(PretrainedConfig):
    """Config for GLiNER-Linker labels encoder (DeBERTa v1 large).

    vLLM-facing: num_hidden_layers=0 for no KV cache.
    DeBERTa v1: encoder_* params for model construction.
    """

    model_type = "gliner_linker"

    def __init__(
        self,
        # vLLM-facing: no KV cache
        num_hidden_layers: int = 0,
        num_attention_heads: int = 1,
        hidden_size: int = 1024,
        # DeBERTa v1 encoder params (from labels_encoder_config)
        vocab_size: int = 50265,
        encoder_hidden_size: int = 1024,
        encoder_num_hidden_layers: int = 24,
        encoder_num_attention_heads: int = 16,
        encoder_intermediate_size: int = 4096,
        encoder_hidden_act: str = "gelu",
        encoder_max_position_embeddings: int = 512,
        encoder_type_vocab_size: int = 0,
        encoder_layer_norm_eps: float = 1e-7,
        encoder_relative_attention: bool = True,
        encoder_max_relative_positions: int = -1,
        encoder_position_biased_input: bool = False,
        encoder_pad_token_id: int = 0,
        encoder_pos_att_type: list = None,
        # Pooling
        pooling_type: str = "MEAN",
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # vLLM-facing
        self.num_hidden_layers = 0
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        # DeBERTa v1 encoder
        self.vocab_size = vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_intermediate_size = encoder_intermediate_size
        self.encoder_hidden_act = encoder_hidden_act
        self.encoder_max_position_embeddings = encoder_max_position_embeddings
        self.encoder_type_vocab_size = encoder_type_vocab_size
        self.encoder_layer_norm_eps = encoder_layer_norm_eps
        self.encoder_relative_attention = encoder_relative_attention
        self.encoder_max_relative_positions = encoder_max_relative_positions
        self.encoder_position_biased_input = encoder_position_biased_input
        self.encoder_pad_token_id = encoder_pad_token_id
        self.encoder_pos_att_type = encoder_pos_att_type or ["c2p", "p2c"]

        # Pooling
        self.pooling_type = pooling_type
        self.normalize = normalize
