from __future__ import annotations

import os
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
PORT = 8000
MODEL_NAME = os.environ.get("GLINER2_VLLM_MODEL", "fastino/gliner2-large-v1")
SERVED_MODEL_NAME = os.environ.get("GLINER2_VLLM_SERVED_MODEL_NAME", MODEL_NAME)
GPU = os.environ.get("GLINER2_VLLM_GPU", "T4")
HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"

hf_cache_vol = modal.Volume.from_name("gliner2-vllm-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("gliner2-vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("vllm/vllm-openai:latest")
    .entrypoint([])
    .uv_sync(str(ROOT))
    .add_local_dir(str(ROOT), remote_path="/root/project")
    .run_commands("cd /root/project && /.uv/uv pip install --system -e .")
)

app = modal.App("gliner2-vllm-factory", image=image)
_SERVER = None


@app.function(
    gpu=GPU,
    timeout=60 * 60,
    scaledown_window=60,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        VLLM_CACHE_PATH: vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    global _SERVER

    from forge.server import ModelServer

    _SERVER = ModelServer(
        name="gliner2-vllm",
        model=MODEL_NAME,
        port=PORT,
        gpu_memory_utilization=0.85,
        max_num_seqs=16,
        dtype="bfloat16",
        trust_remote_code=True,
        served_model_name=SERVED_MODEL_NAME,
        gliner_plugin="deberta_gliner2",
        extra_args=[
            "--runner",
            "pooling",
            "--io-processor-plugin",
            "deberta_gliner2_io",
        ],
    )
    _SERVER.start()
