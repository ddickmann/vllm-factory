from __future__ import annotations

import os
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
PORT = 8000
MODEL_NAME = os.environ.get("GLINER2_VLLM_MODEL", "fastino/gliner2-large-v1")
SERVED_MODEL_NAME = os.environ.get("GLINER2_VLLM_SERVED_MODEL_NAME", MODEL_NAME)
GPU = os.environ.get("GLINER2_VLLM_GPU", "L4")
DTYPE = os.environ.get("GLINER2_VLLM_DTYPE", "auto")
GPU_MEMORY_UTILIZATION = float(os.environ.get("GLINER2_VLLM_GPU_MEM_UTIL", "0.90"))
MAX_NUM_SEQS = int(os.environ.get("GLINER2_VLLM_MAX_NUM_SEQS", "128"))
HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"

hf_cache_vol = modal.Volume.from_name("gliner2-vllm-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("gliner2-vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("vllm/vllm-openai:latest")
    .entrypoint([])
    .add_local_dir(str(ROOT), remote_path="/root/project", copy=True)
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")
    .run_commands("ln -sf $(command -v pip3) /usr/local/bin/pip")
    .run_commands("python3 -m pip install --no-deps --no-build-isolation /root/project")
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
@modal.concurrent(max_inputs=128)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    global _SERVER

    from forge.server import ModelServer

    _SERVER = ModelServer(
        name="gliner2-vllm",
        model=MODEL_NAME,
        port=PORT,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
        dtype=DTYPE,
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
