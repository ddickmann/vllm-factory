"""GLiNER DeBERTa v2 Configuration — encoder-only pooling model.

Uses PretrainedConfig with num_hidden_layers=0 so vLLM skips KV cache
allocation. The actual DeBERTa v2 encoder config is stored as separate
attributes and used by the model to construct DebertaV2EncoderModel.
"""

from transformers import PretrainedConfig


class GLiNERDebertaV2Config(PretrainedConfig):
    """Config for GLiNER with DeBERTa v2 encoder backbone.

    Stores both vLLM-facing params (num_hidden_layers=0 for no KV cache)
    and DeBERTa v2 encoder params for model construction.
    """

    model_type = "gliner_deberta_v2"

    def __init__(
        self,
        # vLLM-facing: no KV cache
        num_hidden_layers: int = 0,
        num_attention_heads: int = 1,
        hidden_size: int = 1024,
        # DeBERTa v2 encoder params
        vocab_size: int = 128004,
        encoder_hidden_size: int = 1024,
        encoder_num_hidden_layers: int = 24,
        encoder_num_attention_heads: int = 16,
        encoder_intermediate_size: int = 4096,
        encoder_hidden_act: str = "gelu",
        encoder_hidden_dropout_prob: float = 0.1,
        encoder_attention_probs_dropout_prob: float = 0.1,
        encoder_max_position_embeddings: int = 512,
        encoder_type_vocab_size: int = 0,
        encoder_layer_norm_eps: float = 1e-7,
        encoder_relative_attention: bool = True,
        encoder_max_relative_positions: int = -1,
        encoder_position_buckets: int = 256,
        encoder_pos_att_type: list = None,
        encoder_share_att_key: bool = True,
        encoder_norm_rel_ebd: str = "layer_norm",
        encoder_position_biased_input: bool = False,
        encoder_pad_token_id: int = 0,
        # GLiNER specific
        gliner_dropout: float = 0.4,
        gliner_hidden_size: int = 512,
        max_width: int = 12,
        class_token_index: int = 128002,
        sep_token_index: int = 128003,
        ent_token: str = "<<ENT>>",
        sep_token: str = "<<SEP>>",
        max_len: int = 384,
        max_neg_type_ratio: int = 1,
        max_types: int = 25,
        subtoken_pooling: str = "first",
        words_splitter_type: str = "whitespace",
        span_mode: str = "markerV0",
        has_rnn: bool = True,
        embed_ent_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # vLLM-facing: ensure no KV cache allocation
        self.num_hidden_layers = 0
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        # DeBERTa v2 encoder params
        self.vocab_size = vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_intermediate_size = encoder_intermediate_size
        self.encoder_hidden_act = encoder_hidden_act
        self.encoder_hidden_dropout_prob = encoder_hidden_dropout_prob
        self.encoder_attention_probs_dropout_prob = encoder_attention_probs_dropout_prob
        self.encoder_max_position_embeddings = encoder_max_position_embeddings
        self.encoder_type_vocab_size = encoder_type_vocab_size
        self.encoder_layer_norm_eps = encoder_layer_norm_eps
        self.encoder_relative_attention = encoder_relative_attention
        self.encoder_max_relative_positions = encoder_max_relative_positions
        self.encoder_position_buckets = encoder_position_buckets
        self.encoder_pos_att_type = encoder_pos_att_type or ["p2c", "c2p"]
        self.encoder_share_att_key = encoder_share_att_key
        self.encoder_norm_rel_ebd = encoder_norm_rel_ebd
        self.encoder_position_biased_input = encoder_position_biased_input
        self.encoder_pad_token_id = encoder_pad_token_id

        # GLiNER head params
        self.gliner_dropout = gliner_dropout
        self.gliner_hidden_size = gliner_hidden_size
        self.max_width = max_width
        self.class_token_index = class_token_index
        self.sep_token_index = sep_token_index
        self.ent_token = ent_token
        self.sep_token = sep_token
        self.max_len = max_len
        self.max_neg_type_ratio = max_neg_type_ratio
        self.max_types = max_types
        self.subtoken_pooling = subtoken_pooling
        self.words_splitter_type = words_splitter_type
        self.span_mode = span_mode
        self.has_rnn = has_rnn
        self.embed_ent_token = embed_ent_token
