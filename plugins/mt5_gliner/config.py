"""GLiNER MT5 Configuration — simple config for encoder-only pooling model.

Does NOT extend MT5Config to avoid num_layers/num_hidden_layers aliasing
that causes vLLM to try allocating KV cache. Instead uses PretrainedConfig
with num_hidden_layers=0 to tell vLLM this is a KV-cache-free model.

The actual T5 encoder config is stored as separate attributes (d_model,
num_layers, etc.) and used by the model to construct the HF T5EncoderModel.
"""

from transformers import PretrainedConfig


class GLiNERMT5Config(PretrainedConfig):
    """Config for GLiNER with MT5 encoder backbone.

    Stores both vLLM-facing params (num_hidden_layers=0 for no KV cache)
    and T5 encoder params (d_model, num_layers, etc.) for model construction.
    """

    model_type = "gliner_mt5"

    def __init__(
        self,
        # vLLM-facing: no KV cache
        num_hidden_layers: int = 0,
        num_attention_heads: int = 1,
        hidden_size: int = 1024,
        # T5 encoder params
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
        max_types: int = 200,
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

        # T5 encoder params (used by model to construct HF T5EncoderModel)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.is_encoder_decoder = is_encoder_decoder
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

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

        # Derived
        self.is_gated_act = feed_forward_proj.startswith("gated")
        self.dense_act_fn = "gelu_new" if "gelu" in feed_forward_proj else "relu"
