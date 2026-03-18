from forge import model_prep
from forge.server import ModelServer


def test_infer_gliner_plugin_from_model_name():
    assert model_prep.infer_gliner_plugin_from_model_name("microsoft/deberta-v3-small") == "deberta_gliner"
    assert model_prep.infer_gliner_plugin_from_model_name("answerdotai/ModernBERT-base") == "mmbert_gliner"
    assert model_prep.infer_gliner_plugin_from_model_name("google/mt5-base") == "mt5_gliner"
    assert model_prep.infer_gliner_plugin_from_model_name("jhu-clsp/mmBERT-base") == "mmbert_gliner"
    assert model_prep.infer_gliner_plugin_from_model_name("jhu-clsp/ettin-encoder-400m") == "mmbert_gliner"
    assert (
        model_prep.infer_gliner_plugin_from_model_name(
            "jhu-clsp/ettin-encoder-68m",
            model_ref="knowledgator/gliner-linker-rerank-v1.0",
        )
        == "modernbert_gliner_rerank"
    )
    assert model_prep.infer_gliner_plugin_from_model_name("unknown/model") is None


def test_prepare_model_for_vllm_if_needed_passthrough_for_non_gliner(monkeypatch):
    monkeypatch.setattr(model_prep, "list_repo_files", lambda _: ["config.json", "model.safetensors"])
    out = model_prep.prepare_model_for_vllm_if_needed("org/regular-model")
    assert out == "org/regular-model"


def test_prepare_model_for_vllm_if_needed_gliner_auto_prep(monkeypatch):
    monkeypatch.setattr(model_prep, "list_repo_files", lambda _: ["gliner_config.json", "pytorch_model.bin"])
    monkeypatch.setattr(model_prep, "_download_file", lambda *_: "/tmp/fake-gliner-config.json")
    monkeypatch.setattr(model_prep, "_read_json", lambda _: {"model_name": "microsoft/deberta-v3-small"})

    called = {}

    def _fake_prepare(hf_model_id, plugin, output_dir=None, force=False):
        called["hf_model_id"] = hf_model_id
        called["plugin"] = plugin
        called["output_dir"] = output_dir
        called["force"] = force
        return "/tmp/prepared-model"

    monkeypatch.setattr(model_prep, "prepare_gliner_model", _fake_prepare)

    out = model_prep.prepare_model_for_vllm_if_needed("urchade/gliner_small-v2.1")
    assert out == "/tmp/prepared-model"
    assert called["hf_model_id"] == "urchade/gliner_small-v2.1"
    assert called["plugin"] == "deberta_gliner"


def test_model_server_resolves_gliner_model_before_serve(monkeypatch):
    monkeypatch.setattr(
        "forge.server.prepare_model_for_vllm_if_needed",
        lambda model_ref, plugin=None: "/tmp/gliner-prepared",
    )
    monkeypatch.setattr(
        "forge.server.get_gliner_base_model_name",
        lambda model_ref: "microsoft/deberta-v3-small",
    )

    server = ModelServer(name="gliner", model="urchade/gliner_small-v2.1", port=8123)
    server._resolve_model_for_server()
    assert server.model == "/tmp/gliner-prepared"
    assert server.tokenizer == "microsoft/deberta-v3-small"


def test_model_server_build_command_skips_task_by_default():
    server = ModelServer(name="gliner", model="/tmp/gliner-prepared", port=8123)
    cmd = server._build_command()
    assert "--task" not in cmd


def test_get_gliner_base_model_name(monkeypatch):
    monkeypatch.setattr(model_prep, "list_repo_files", lambda _: ["gliner_config.json", "pytorch_model.bin"])
    monkeypatch.setattr(model_prep, "_download_file", lambda *_: "/tmp/fake-gliner-config.json")
    monkeypatch.setattr(model_prep, "_read_json", lambda _: {"model_name": "microsoft/deberta-v3-small"})
    assert model_prep.get_gliner_base_model_name("urchade/gliner_small-v2.1") == "microsoft/deberta-v3-small"
