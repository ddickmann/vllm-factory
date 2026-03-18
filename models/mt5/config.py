from transformers import MT5Config


class GLiNERMT5Config(MT5Config):
    model_type = "gliner_mt5"

    def __init__(
        self,
        # MT5 base config
        vocab_size: int = 250112,
        d_model: int = 1024,
        d_kv: int = 64,
        d_ff: int = 2816,
        num_layers: int = 24,
        num_decoder_layers: int = 24,
        num_heads: int = 16,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 1.0,
        feed_forward_proj: str = "gated-gelu",
        is_encoder_decoder: bool = False,
        use_cache: bool = False,
        tie_word_embeddings: bool = False,
        # GLiNER specific
        gliner_dropout: float = 0.3,
        gliner_hidden_size: int = 768,
        max_width: int = 12,
        class_token_index: int = 250100,
        sep_token_index: int = 250101,
        ent_token: str = "<<ENT>>",
        sep_token: str = "<<SEP>>",
        max_len: int = 1024,
        max_neg_type_ratio: int = 1,
        max_types: int = 30,
        subtoken_pooling: str = "first",
        words_splitter_type: str = "whitespace",
        span_mode: str = "markerV0",
        has_rnn: bool = True,
        embed_ent_token: bool = True,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_factor=initializer_factor,
            feed_forward_proj=feed_forward_proj,
            is_encoder_decoder=is_encoder_decoder,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

        # GLiNER-specific attributes
        self.gliner_dropout = gliner_dropout
        self.max_width = max_width
        self.class_token_index = class_token_index
        self.sep_token_index = sep_token_index
        self.gliner_hidden_size = gliner_hidden_size
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

        # For is_gated_act check in our model code
        self.is_gated_act = feed_forward_proj.startswith("gated")
        self.dense_act_fn = "gelu_new" if "gelu" in feed_forward_proj else "relu"

        # Auto mapping for transformers
        self.auto_map = {
            "AutoConfig": "modeling.GLiNERMT5Config",
            "AutoModel": "modeling.GLiNERMT5Model",
        }
