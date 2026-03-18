"""Config for GLiNER L4 reranker (ModernBERT / ettin backbone + GLiNER heads)."""

from __future__ import annotations

from transformers import ModernBertConfig

# Matches vLLM / transformers nested RoPE layout (sliding + full attention).
_ROPE_NEST_KEYS = frozenset(
    {"full_attention", "sliding_attention", "chunked_attention", "linear_attention"}
)


class GLiNERRerankConfig(ModernBertConfig):
    """HF config for vLLM: ModernBERT encoder + GLiNER uni-encoder rerank metadata.

    ``encoder_*`` fields come from the checkpoint's ``gliner_config.json`` →
    ``encoder_config``. Additional GLiNER fields match ``GLiNER`` root config.
    """

    model_type = "modernbert_gliner_rerank"

    def __init__(
        self,
        class_token_index: int = 50368,
        embed_ent_token: bool = True,
        gliner_hidden_size: int = 768,
        gliner_max_len: int = 2048,
        gliner_dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_token_index = class_token_index
        self.embed_ent_token = embed_ent_token
        self.gliner_hidden_size = gliner_hidden_size
        self.gliner_max_len = gliner_max_len
        self.gliner_dropout = gliner_dropout

    def __getattribute__(self, key: str):
        # ``PretrainedConfig.__getattribute__`` maps ``rope_theta`` → ``global_rope_theta``
        # before properties run. Intercept ``rope_theta`` so vLLM's ``patch_rope_parameters``
        # (Transformers v4) does not inject a sibling key into nested ``rope_parameters``.
        if key == "rope_theta":
            try:
                rp = object.__getattribute__(self, "rope_parameters")
            except AttributeError:
                rp = None
            if isinstance(rp, dict) and set(rp.keys()).issubset(_ROPE_NEST_KEYS):
                return None
        return super().__getattribute__(key)
