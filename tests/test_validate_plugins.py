from forge import validate_plugins


def test_plugin_matrix_has_all_12_plugins():
    assert len(validate_plugins.PLUGIN_MATRIX) == 12


def test_classify_probe_output_server_started():
    status, reason = validate_plugins._classify_probe_output(
        exit_code=0,
        output="INFO Application startup complete. Uvicorn running on http://0.0.0.0:8123",
    )
    assert status == "pass"
    assert reason == "server-started"


def test_classify_probe_output_runtime_incompatible():
    status, reason = validate_plugins._classify_probe_output(
        exit_code=1,
        output="[PREFLIGHT] Incompatible torch runtime for local vLLM CPU startup",
    )
    assert status == "fail"
    assert reason == "runtime-incompatible"
