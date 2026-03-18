"""
GLiNER ModernBERT Configuration.

Extends PretrainedConfig (NOT ModernBertConfig) to prevent vLLM from
allocating KV cache for this encoder-only pooling model.

Stores the real ModernBERT config fields (hidden_size, encoder_num_layers,
num_attention_heads, etc.) so the model can reconstruct a full ModernBertConfig
internally for the encoder backbone.
"""

from transformers import PretrainedConfig


class GLiNERModernBertConfig(PretrainedConfig):
    """Configuration for GLiNER with custom ModernBERT backbone.

    Uses PretrainedConfig as base (not ModernBertConfig) so vLLM sees
    num_hidden_layers=0 and doesn't allocate KV cache. The real
    encoder architecture is configured via encoder_* fields.
    """

    model_type = "gliner_mmbert"

    def __init__(
        self,
        # vLLM sees these — must be 0 to disable KV cache
        num_hidden_layers: int = 0,
        num_attention_heads: int = 1,
        # Real encoder architecture (used by model.py)
        encoder_num_layers: int = 22,
        encoder_num_attention_heads: int = 12,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        vocab_size: int = 256002,
        max_position_embeddings: int = 8192,
        hidden_activation: str = "gelu",
        norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        local_attention: int = 128,
        global_attn_every_n_layers: int = 3,
        global_rope_theta: float = 160000.0,
        local_rope_theta: float = 10000.0,
        # GLiNER specific
        gliner_dropout: float = 0.3,
        max_width: int = 12,
        width_emb_dim: int = 128,
        class_token_index: int = 256000,
        sep_token_index: int = 256001,
        ent_token: str = "<<ENT>>",
        sep_token: str = "<<SEP>>",
        max_len: int = 2048,
        max_neg_type_ratio: int = 1,
        max_types: int = 200,
        subtoken_pooling: str = "first",
        words_splitter_type: str = "whitespace",
        has_rnn: bool = True,
        embed_ent_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # vLLM-facing (triggers 0 KV cache layers)
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # Real encoder architecture
        self.encoder_num_layers = encoder_num_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_activation = hidden_activation
        self.norm_eps = norm_eps
        self.pad_token_id = pad_token_id
        self.local_attention = local_attention
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.global_rope_theta = global_rope_theta
        self.local_rope_theta = local_rope_theta
        # GLiNER
        self.gliner_dropout = gliner_dropout
        self.max_width = max_width
        self.width_emb_dim = width_emb_dim
        self.class_token_index = class_token_index
        self.sep_token_index = sep_token_index
        self.ent_token = ent_token
        self.sep_token = sep_token
        self.max_len = max_len
        self.max_neg_type_ratio = max_neg_type_ratio
        self.max_types = max_types
        self.subtoken_pooling = subtoken_pooling
        self.words_splitter_type = words_splitter_type
        self.has_rnn = has_rnn
        self.embed_ent_token = embed_ent_token
        self.auto_map = {
            "AutoConfig": "config.GLiNERModernBertConfig",
            "AutoModel": "model.GLiNERModernBertModel",
        }
