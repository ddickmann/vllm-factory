"""
ColLFM2 vLLM Server — standalone server launcher.

Uses forge.server.ModelServer to start a vLLM serve process with all
engine args tuned for LFM2-VL multimodal pooling.

Mirrors the engine configuration from handler.py / superpod processor.py:
  - runner=pooling (not task=embed)
  - enforce_eager=True
  - skip_mm_profiling (via extra_args)
  - mm_processor_cache_gb=1 (via extra_args)
  - limit_mm_per_prompt={"image":1} (via extra_args)
  - no prefix caching / no chunked prefill

Usage:
    python plugins/collfm2/serve.py \\
        --model VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1 \\
        --port 8000

Then in another terminal:
    python plugins/collfm2/benchmark.py \\
        --model VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1

Or with curl:
    curl -s http://localhost:8000/v1/models | python -m json.tool
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
logger = logging.getLogger("collfm2.serve")


def main():
    p = argparse.ArgumentParser(description="Start vLLM server for ColLFM2")
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--quantization", default=None)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    args = p.parse_args()

    # Import here (after env setup) to avoid initializing CUDA prematurely
    from forge.server import ModelServer

    # Install plugin so vLLM can find ColLFM2 model type
    try:
        import collfm2  # noqa: F401 — triggers register()
        logger.info("ColLFM2 plugin registered")
    except ImportError:
        logger.warning(
            "collfm2 package not installed — run: pip install -e plugins/collfm2/"
        )

    server = ModelServer(
        name="collfm2",
        model=args.model,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        quantization=args.quantization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        trust_remote_code=True,
        task="token_embed",    # vLLM 0.15.1: multi-vector retrieval task
        extra_args=[
            "--skip-mm-profiling",           # avoids OOM during VLM memory profiling
            "--mm-processor-cache-gb", "1",  # reduce multimodal cache (default: 4GB)
            "--limit-mm-per-prompt", '{"image": 1}',  # one image per request (ColPali)
            "--uvicorn-log-level", "warning",
        ],
    )

    logger.info(f"Starting ColLFM2 server on port {args.port}...")
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

    logger.info(f"ColLFM2 server ready at http://localhost:{args.port}")
    logger.info("Press Ctrl+C to stop.")

    # Block until the server process dies
    try:
        server.process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
