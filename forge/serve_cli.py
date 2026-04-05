"""CLI for multi-instance vLLM serving: ``vllm-factory-serve``.

When ``--num-instances 1`` (the default) the command launches a single
:class:`~forge.server.ModelServer` — identical to the existing behaviour.

When ``--num-instances N`` (N > 1) it launches N backends behind a thin
reverse-proxy dispatcher via :class:`~forge.multi_instance.MultiInstanceServer`.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

logger = logging.getLogger("vllm-factory.serve")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vllm-factory-serve",
        description=(
            "Serve a vLLM model with optional multi-instance scaling (beta). "
            "Extra flags after '--' are forwarded to each vllm serve backend."
        ),
    )
    parser.add_argument(
        "model",
        help="HuggingFace model ID or local model path",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=1,
        help="Number of vLLM backend instances (default: 1)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Per-backend max concurrent requests / max-num-seqs (default: 32)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="User-facing port (default: 8000)",
    )
    parser.add_argument(
        "--port-start",
        type=int,
        default=9100,
        help="First internal backend port when num-instances > 1 (default: 9100)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype (default: auto)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph compilation",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code in model repos (default: True)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Per-instance GPU memory utilisation (auto-scaled when omitted for multi-instance)",
    )
    parser.add_argument(
        "--io-processor-plugin",
        default=None,
        help="vLLM IOProcessor plugin name",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="vLLM task (e.g. 'plugin')",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model sequence length",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of batched tokens per iteration",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Explicit tokenizer path or name",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Model name exposed to clients",
    )
    parser.add_argument(
        "--cuda-devices",
        default="0",
        help="CUDA_VISIBLE_DEVICES for the backend(s) (default: '0')",
    )

    args, extra = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    extra_args = [a for a in extra if a != "--"]
    if args.io_processor_plugin:
        extra_args.extend(["--io-processor-plugin", args.io_processor_plugin])

    model_kwargs = dict(
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        task=args.task,
        tokenizer=args.tokenizer,
        served_model_name=args.served_model_name,
        cuda_devices=args.cuda_devices,
    )

    num = max(1, args.num_instances)

    if num == 1:
        _run_single(args, extra_args, model_kwargs)
    else:
        _run_multi(args, num, extra_args, model_kwargs)


def _run_single(
    args: argparse.Namespace,
    extra_args: list[str],
    model_kwargs: dict,
) -> None:
    """Single-instance path — delegates directly to ModelServer."""
    from forge.server import ModelServer

    gpu_util = args.gpu_memory_utilization or 0.90
    server = ModelServer(
        name="default",
        model=args.model,
        port=args.port,
        gpu_memory_utilization=gpu_util,
        max_num_seqs=args.max_batch_size,
        extra_args=extra_args,
        **model_kwargs,
    )
    logger.info(f"Starting single-instance server on port {args.port}")

    def _shutdown(signum, frame):
        logger.info("Shutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server.start()

    logger.info(f"Server ready on port {args.port}. Press Ctrl+C to stop.")
    try:
        server.process.wait()
    except (KeyboardInterrupt, AttributeError):
        pass
    finally:
        server.stop()


def _run_multi(
    args: argparse.Namespace,
    num_instances: int,
    extra_args: list[str],
    model_kwargs: dict,
) -> None:
    """Multi-instance path — launches backends + dispatcher."""
    from forge.multi_instance import MultiInstanceServer

    logger.info(
        f"[beta] Multi-instance mode: {num_instances} backends, "
        f"max_bs={args.max_batch_size}"
    )

    multi = MultiInstanceServer(
        model=args.model,
        num_instances=num_instances,
        max_bs=args.max_batch_size,
        port=args.port,
        port_start=args.port_start,
        extra_args=extra_args,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **model_kwargs,
    )
    multi.start()


if __name__ == "__main__":
    main()
