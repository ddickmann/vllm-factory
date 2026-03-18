"""
ColQwen3 vLLM Server — standalone server launcher.

Uses forge.server.ModelServer to start a vLLM serve process with all
engine args tuned for Qwen3-VL multimodal pooling.

The slow Qwen2VL image processor is unconditionally enforced in
ColQwen3ProcessingInfo (model.py) for parity with the reference
implementation (cosine >= 0.99).

Usage:
    PYTHONPATH=/workspace/vllm-factory python plugins/colqwen3/serve.py \\
        --model VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1 \\
        --port 8000

Then in another terminal:
    python plugins/colqwen3/benchmark.py \\
        --model VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("colqwen3.serve")


def main():
    p = argparse.ArgumentParser(description="Start vLLM server for ColQwen3")
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--quantization", default=None)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    args = p.parse_args()

    # Install plugin so vLLM can find ColQwen3 model type
    try:
        import colqwen3  # noqa: F401 — triggers register()
        logger.info("ColQwen3 plugin registered")
    except ImportError:
        logger.warning(
            "colqwen3 package not installed — run: pip install -e plugins/colqwen3/"
        )

    from forge.server import ModelServer

    server = ModelServer(
        name="colqwen3",
        model=args.model,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        quantization=args.quantization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,    # not beneficial for doc retrieval
        enable_chunked_prefill=False,   # causes issues with multimodal inputs
        trust_remote_code=True,
        runner="pooling",               # vLLM 0.15.x embedding mode
        extra_args=[
            "--skip-mm-profiling",           # avoids OOM during VLM memory profiling
            "--mm-processor-cache-gb", "1",  # reduce multimodal cache (default: 4GB)
            "--limit-mm-per-prompt", '{"image": 1}',  # one image per request (ColPali)
            "--uvicorn-log-level", "warning",
            # Slow image processor is patched at Python import time in __init__.py
        ],
    )

    logger.info(f"Starting ColQwen3 server on port {args.port}...")
    logger.info(f"  model: {args.model}")
    logger.info(f"  gpu_memory_utilization: {args.gpu_memory_utilization}")
    logger.info(f"  max_model_len: {args.max_model_len}")
    logger.info(f"  max_num_seqs: {args.max_num_seqs}")

    def _shutdown(sig, frame):
        logger.info("Shutdown requested, stopping server...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.start()

    logger.info(f"ColQwen3 server ready at http://localhost:{args.port}")
    logger.info("Press Ctrl+C to stop.")

    try:
        server.process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
