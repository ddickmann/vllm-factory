"""
ModernColBERT vLLM Server

Launches a vLLM API server for the ModernColBERT plugin. Sets PYTHONPATH
to include both plugins/ (for moderncolbert module) and models/ (for the
custom ModernBERT encoder), matching the superpod handler configuration.

Usage:
    python plugins/moderncolbert/serve.py \
        --model VAGOsolutions/SauerkrautLM-Multi-ModernColBERT \
        --port 8100
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

# Base of the vllm-factory repo
REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGINS_DIR = str(REPO_ROOT / "plugins")
MODELS_DIR = str(REPO_ROOT / "models")


def build_python_path() -> str:
    """Build PYTHONPATH that includes both plugins/ and models/."""
    existing = os.environ.get("PYTHONPATH", "")
    paths = [p for p in existing.split(":") if p]
    for d in [PLUGINS_DIR, MODELS_DIR]:
        if d not in paths:
            paths.insert(0, d)
    return ":".join(paths)


def main():
    parser = argparse.ArgumentParser(description="ModernColBERT vLLM API server")
    parser.add_argument("--model", default="VAGOsolutions/SauerkrautLM-Multi-ModernColBERT")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = build_python_path()
    # tokenizer_config.json has model_max_length=299 but actual model supports 8192
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    # Prevent tokenizer parallelism deadlocks in async contexts
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--runner", "pooling",
        "--trust-remote-code",
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        # Critical: limit profiling batch to max_model_len to avoid OOM
        # from the seq_len x seq_len attention mask allocation during warmup
        "--max-num-batched-tokens", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--uvicorn-log-level", "warning",
    ]

    print()
    print("=" * 70)
    print("ModernColBERT vLLM Server")
    print(f"  model:  {args.model}")
    print(f"  listen: {args.host}:{args.port}")
    print(f"  dtype:  {args.dtype}")
    print(f"  max_model_len: {args.max_model_len}")
    print("=" * 70)
    print(f"  PYTHONPATH: {env['PYTHONPATH']}")
    print()

    proc = subprocess.Popen(cmd, env=env)

    def _shutdown(sig, frame):
        print(f"\n[serve] Caught signal {sig}, shutting down...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    proc.wait()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
