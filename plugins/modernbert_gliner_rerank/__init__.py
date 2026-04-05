"""GLiNER L4 reranker (ModernBERT + LSTM + scorer) for vLLM pooling / LLM.embed.

**Production-ready.** Uses a custom ModernBERT encoder for numerical parity
with vanilla GLiNER. See ``docs/gliner/L4_PARITY.md``.

The HuggingFace checkpoint ships ``gliner_config.json`` but no root ``config.json``.
This plugin materializes a vLLM- and ``GLiNERRerankConfig``-compatible ``config.json``
under ``_model_cache/`` and symlinks weights + tokenizer files from the HF snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import GLiNERRerankConfig
from .model import GLiNERRerankModel

HF_MODEL_ID = "knowledgator/gliner-linker-rerank-v1.0"
PLUGIN_DIR = Path(__file__).parent
MODEL_CACHE_DIR = PLUGIN_DIR / "_model_cache"

# Strip HF encoder JSON noise; keys not accepted by ``ModernBertConfig`` / transformers.
_ENCODER_DROP_KEYS = frozenset(
    {
        "model_type",
        "architectures",
        "dtype",
        "id2label",
        "label2id",
        "problem_type",
        "torchscript",
        "tf_legacy_loss",
        "prefix",
        "pruned_heads",
        "_name_or_path",
        "task_specific_params",
        "decoder_start_token_id",
        "is_encoder_decoder",
        "is_decoder",
        "output_attentions",
        "output_hidden_states",
        "return_dict",
        "finetuning_task",
    }
)

HF_FILES_TO_LINK = [
    "pytorch_model.bin",
    "gliner_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

# vLLM ``patch_rope_parameters`` (Transformers v4 path) does
# ``config.rope_parameters["rope_theta"] = rope_theta``, which breaks **nested**
# ModernBERT ``rope_parameters`` (full_attention / sliding_attention): extra keys
# make ``is_rope_parameters_nested`` false and validation raises "rope_type" missing.
_ROPE_NEST_KEYS = frozenset(
    {"full_attention", "sliding_attention", "chunked_attention", "linear_attention"}
)


def _sanitize_vllm_rope_config_dict(cfg_dict: dict) -> None:
    rp = cfg_dict.get("rope_parameters")
    if not isinstance(rp, dict) or not set(rp.keys()).issubset(_ROPE_NEST_KEYS):
        return
    for k in (
        "rope_theta",
        "partial_rotary_factor",
        "rotary_pct",
        "rotary_emb_fraction",
        "rotary_emb_base",
        "original_max_position_embeddings",
    ):
        cfg_dict.pop(k, None)


def _config_dict_from_gliner_json(hf_path: Path) -> dict:
    raw = json.loads((hf_path / "gliner_config.json").read_text())
    enc = {k: v for k, v in raw["encoder_config"].items() if k not in _ENCODER_DROP_KEYS}
    return {
        **enc,
        "class_token_index": raw["class_token_index"],
        "embed_ent_token": raw.get("embed_ent_token", True),
        "gliner_hidden_size": raw["hidden_size"],
        "gliner_max_len": raw.get("max_len", 2048),
        "gliner_dropout": raw.get("dropout", 0.3),
    }


def prepare_model_dir() -> str:
    """Create ``_model_cache`` with ``config.json`` and symlinks to HF weights/tokenizer."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        hf_dir = snapshot_download(HF_MODEL_ID)
    except Exception as e:
        raise RuntimeError(
            f"Cannot download or locate {HF_MODEL_ID}. "
            f"Run `huggingface-cli download {HF_MODEL_ID}` first.\nError: {e}"
        ) from e

    hf_path = Path(hf_dir)
    cfg = GLiNERRerankConfig(**_config_dict_from_gliner_json(hf_path))
    cfg_dict = cfg.to_dict()
    cfg_dict["model_type"] = "modernbert_gliner_rerank"
    cfg_dict["architectures"] = ["GLiNERRerankModel"]
    _sanitize_vllm_rope_config_dict(cfg_dict)

    (MODEL_CACHE_DIR / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    for filename in HF_FILES_TO_LINK:
        src = hf_path / filename
        dst = MODEL_CACHE_DIR / filename
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())

    return str(MODEL_CACHE_DIR)


def get_model_path() -> str:
    return prepare_model_dir()


def register() -> None:
    from forge.registration import register_plugin

    register_plugin(
        "modernbert_gliner_rerank",
        GLiNERRerankConfig,
        "GLiNERRerankModel",
        GLiNERRerankModel,
    )


register()

__all__ = ["GLiNERRerankModel", "GLiNERRerankConfig", "get_model_path", "HF_MODEL_ID"]
