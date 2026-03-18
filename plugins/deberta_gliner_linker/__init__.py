"""GLiNER-Linker Plugin — registers the bi-encoder entity linker with vLLM.

**Production-ready.** Passes end-to-end recall-gated parity testing in bfloat16.
See ``docs/gliner/README.md``.

Handles two critical compatibility issues at registration time:

1. **Missing config.json**: The HuggingFace model only provides
   `gliner_config.json`, but vLLM requires a standard `config.json`.
   We auto-generate one in a local model directory.

2. **Tokenizer class mismatch**: The model uses `tokenizer_class:
   "TokenizersBackend"` which `AutoTokenizer` doesn't recognize.
   We patch it to `PreTrainedTokenizerFast`.

The auto-generated model directory is created under the plugin's own
directory as `_model_cache/`, containing symlinks to the HF-cached
files plus our custom config.json and tokenizer_config.json.
"""

import json
from pathlib import Path

from .config import GLiNERLinkerConfig
from .model import GLiNERLinkerModel

HF_MODEL_ID = "knowledgator/gliner-linker-large-v1.0"
PLUGIN_DIR = Path(__file__).parent
MODEL_CACHE_DIR = PLUGIN_DIR / "_model_cache"

# The config.json we generate for vLLM compatibility.
# Derived from gliner_config.json → encoder_config section.
VLLM_CONFIG = {
    "model_type": "gliner_linker",
    "architectures": ["GLiNERLinkerModel"],
    "num_hidden_layers": 0,
    "num_attention_heads": 1,
    "hidden_size": 1024,
    "vocab_size": 50265,
    "encoder_hidden_size": 1024,
    "encoder_num_hidden_layers": 24,
    "encoder_num_attention_heads": 16,
    "encoder_intermediate_size": 4096,
    "encoder_hidden_act": "gelu",
    "encoder_max_position_embeddings": 512,
    "encoder_type_vocab_size": 0,
    "encoder_layer_norm_eps": 1e-07,
    "encoder_relative_attention": True,
    "encoder_max_relative_positions": -1,
    "encoder_position_biased_input": False,
    "encoder_pad_token_id": 0,
    "encoder_pos_att_type": ["c2p", "p2c"],
    "pooling_type": "MEAN",
    "normalize": False,
}

# Fixed tokenizer config (replaces TokenizersBackend with PreTrainedTokenizerFast)
TOKENIZER_CONFIG = {
    "add_prefix_space": True,
    "backend": "tokenizers",
    "bos_token": "[CLS]",
    "clean_up_tokenization_spaces": False,
    "cls_token": "[CLS]",
    "do_lower_case": False,
    "eos_token": "[SEP]",
    "errors": "replace",
    "is_local": True,
    "mask_token": "[MASK]",
    "model_max_length": 1000000000000000019884624838656,
    "model_specific_special_tokens": {},
    "pad_token": "[PAD]",
    "sep_token": "[SEP]",
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "[UNK]",
    "vocab_type": "gpt2",
}

# Files to symlink from the HF cache (everything except what we override)
HF_FILES_TO_LINK = [
    "pytorch_model.bin",
    "gliner_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
]


def prepare_model_dir() -> str:
    """Auto-generate local model directory with vLLM-compatible config.

    Creates `_model_cache/` under the plugin directory with:
    - config.json (custom, for vLLM)
    - tokenizer_config.json (fixed tokenizer class)
    - Symlinks to all other files in the HF cache

    Returns:
        Absolute path to the prepared model directory.

    Raises:
        RuntimeError: If the HF model files cannot be found/downloaded.
    """
    if MODEL_CACHE_DIR.exists() and (MODEL_CACHE_DIR / "config.json").exists():
        # Already prepared — verify weights symlink is valid
        weights = MODEL_CACHE_DIR / "pytorch_model.bin"
        if weights.exists():
            return str(MODEL_CACHE_DIR)

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download/locate HF model files
    try:
        from huggingface_hub import snapshot_download
        hf_dir = snapshot_download(HF_MODEL_ID)
    except Exception as e:
        raise RuntimeError(
            f"Cannot download or locate {HF_MODEL_ID}. "
            f"Run `huggingface-cli download {HF_MODEL_ID}` first.\n"
            f"Error: {e}"
        ) from e

    hf_path = Path(hf_dir)

    # Write our custom config files
    (MODEL_CACHE_DIR / "config.json").write_text(
        json.dumps(VLLM_CONFIG, indent=2)
    )
    (MODEL_CACHE_DIR / "tokenizer_config.json").write_text(
        json.dumps(TOKENIZER_CONFIG, indent=2)
    )

    # Symlink remaining files from HF cache
    for filename in HF_FILES_TO_LINK:
        src = hf_path / filename
        dst = MODEL_CACHE_DIR / filename
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())

    return str(MODEL_CACHE_DIR)


def get_model_path() -> str:
    """Get the path to the prepared model directory.

    Returns the auto-generated `_model_cache/` path. If not yet prepared,
    calls `prepare_model_dir()` to create it.
    """
    return prepare_model_dir()


def register() -> None:
    """Register GLiNER-Linker with vLLM and HuggingFace."""
    from forge.registration import register_plugin

    # Forward collator attention masks through vLLM pooling → model.forward (batched embed).
    from plugins.deberta_gliner_linker.vllm_pooling_attention_mask import (
        apply_pooling_attention_mask_patch,
    )

    apply_pooling_attention_mask_patch()
    register_plugin("gliner_linker", GLiNERLinkerConfig,
                     "GLiNERLinkerModel", GLiNERLinkerModel)


register()

__all__ = ["GLiNERLinkerModel", "GLiNERLinkerConfig", "get_model_path"]
