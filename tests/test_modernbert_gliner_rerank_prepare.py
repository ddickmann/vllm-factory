"""CPU-only checks: model cache layout and HF config merge."""

from __future__ import annotations

import json
from pathlib import Path

from plugins.modernbert_gliner_rerank import prepare_model_dir
from plugins.modernbert_gliner_rerank.config import GLiNERRerankConfig


def test_prepare_model_dir_writes_config_and_weights_symlink():
    model_dir = Path(prepare_model_dir())
    assert model_dir.is_dir()
    cfg_path = model_dir / "config.json"
    assert cfg_path.is_file()
    raw = json.loads(cfg_path.read_text())
    assert raw.get("model_type") == "modernbert_gliner_rerank"
    assert raw.get("architectures") == ["GLiNERRerankModel"]
    assert raw.get("hidden_size") == 512
    assert raw.get("gliner_hidden_size") == 768
    w = model_dir / "pytorch_model.bin"
    assert w.exists()


def test_gligner_rerank_config_roundtrip_from_cache():
    model_dir = prepare_model_dir()
    cfg = GLiNERRerankConfig.from_pretrained(model_dir)
    assert cfg.model_type == "modernbert_gliner_rerank"
    assert cfg.vocab_size == 50370
