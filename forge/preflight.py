from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger("vllm-factory.preflight")


def require_native_io_path() -> None:
    """Verify that vLLM supports the native IOProcessor path.

    vLLM >= 0.19 provides this natively. If a pre-0.19 vLLM is installed,
    this will raise with clear instructions.
    """
    from vllm_factory.compat.vllm_capabilities import detect

    caps = detect()
    if caps.has_io_processor_interface and caps.has_io_processor_response:
        logger.info(
            "[PREFLIGHT] Native IOProcessor path available (vLLM %s).",
            caps.version,
        )
        return

    raise RuntimeError(
        f"[PREFLIGHT] vLLM {caps.version or 'UNKNOWN'} does not support the "
        "native IOProcessor path. vllm-factory requires vLLM >= 0.19. "
        "Upgrade with: pip install 'vllm>=0.19'"
    )


def require_runtime_compatibility() -> None:
    """Fail fast for known local runtime incompatibilities.

    This check only matters for CPU-only / macOS dev setups where
    torch.ops._C_utils.init_cpu_threads_env may be missing.
    On CUDA-enabled systems it is always skipped.
    """
    if os.getenv("VLLM_FACTORY_SKIP_RUNTIME_COMPAT_CHECK") == "1":
        return

    try:
        import torch
    except Exception as e:
        raise RuntimeError(
            "[PREFLIGHT] Failed to import torch for runtime compatibility checks."
        ) from e

    if torch.cuda.is_available():
        return

    c_utils = getattr(torch.ops, "_C_utils", None)
    has_init_cpu_threads_env = bool(
        c_utils is not None and hasattr(c_utils, "init_cpu_threads_env")
    )
    if not has_init_cpu_threads_env:
        raise RuntimeError(
            "[PREFLIGHT] Incompatible torch runtime for local vLLM CPU startup: "
            "torch.ops._C_utils.init_cpu_threads_env is missing. "
            "Use a compatible torch/vLLM pair or run on a GPU-backed Linux runtime. "
            "Set VLLM_FACTORY_SKIP_RUNTIME_COMPAT_CHECK=1 to bypass."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM Factory startup preflight checks")
    parser.add_argument(
        "--require-native-io",
        action="store_true",
        help="Verify that vLLM supports native IOProcessor path.",
    )
    parser.add_argument(
        "--require-runtime-compat",
        action="store_true",
        help="Verify local torch/vLLM runtime compatibility before startup.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run the environment diagnostics (doctor).",
    )
    args = parser.parse_args()

    if args.doctor:
        from vllm_factory.compat.doctor import run_doctor
        run_doctor()
        return

    if args.require_native_io:
        require_native_io_path()
    if args.require_runtime_compat:
        require_runtime_compatibility()
    if not args.require_native_io and not args.require_runtime_compat:
        logger.info("[PREFLIGHT] No checks selected.")


if __name__ == "__main__":
    main()
