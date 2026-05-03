"""GLiNER2 ModernBERT Configuration — ModernBERT encoder + GLiNER2 pooler."""

from transformers import PretrainedConfig


class GLiNER2ModernBertConfig(PretrainedConfig):
    """Configuration for ModernBERT + GLiNER2 vLLM plugin.

    Uses PretrainedConfig as base (not ModernBertConfig) so vLLM sees
    num_hidden_layers=0 and doesn't allocate KV cache. The real
    encoder architecture is configured via encoder_* fields.
    """

    model_type = "mmbert_gliner2"

    def __init__(
        self,
        # vLLM-facing: no KV cache
        num_hidden_layers: int = 0,
        num_attention_heads: int = 1,
        # ModernBERT encoder params
        hidden_size: int = 384,
        encoder_num_layers: int = 22,
        encoder_num_attention_heads: int = 6,
        intermediate_size: int = 1152,
        vocab_size: int = 256010,
        max_position_embeddings: int = 8192,
        hidden_activation: str = "gelu",
        norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        local_attention: int = 128,
        global_attn_every_n_layers: int = 3,
        global_rope_theta: float = 160000.0,
        local_rope_theta: float = 160000.0,
        layer_types: list[str] | None = None,
        rope_parameters: dict | None = None,
        # GLiNER2 head params
        max_width: int = 12,
        counting_layer: str = "count_lstm_v2",
        token_pooling: str = "first",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hidden_layers = 0
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
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
        self.rope_parameters = rope_parameters or {
            "full_attention": {
                "rope_theta": global_rope_theta,
                "rope_type": "default",
            },
            "sliding_attention": {
                "rope_theta": local_rope_theta,
                "rope_type": "default",
            },
        }
        self.layer_types = layer_types or [
            "full_attention" if i % global_attn_every_n_layers == 0 else "sliding_attention"
            for i in range(encoder_num_layers)
        ]
        self.max_width = max_width
        self.counting_layer = counting_layer
        self.token_pooling = token_pooling
