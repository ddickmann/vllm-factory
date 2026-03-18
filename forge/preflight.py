from __future__ import annotations

import argparse
import os

from forge.patches import pooling_extra_kwargs as pooling_patch


def require_pooling_patch_ready() -> None:
    """Fail fast unless pooling patch verification passes.

    Optional behavior:
    - Set VLLM_FACTORY_AUTO_APPLY_POOLING_PATCH=1 to auto-apply before verify.
    """
    pooling_patch.ensure_supported_vllm_version(strict=True)

    auto_apply = os.getenv("VLLM_FACTORY_AUTO_APPLY_POOLING_PATCH") == "1"
    if auto_apply:
        print("[PREFLIGHT] Auto-apply enabled, applying pooling patch before verify.")
        pooling_patch.apply_patch()
        return

    ok = pooling_patch.verify_patch()
    if ok:
        print("[PREFLIGHT] Pooling patch verification passed.")
        return

    raise RuntimeError(
        "[PREFLIGHT] Pooling patch verification failed. "
        "Run `python -m forge.patches.pooling_extra_kwargs` "
        "or set VLLM_FACTORY_AUTO_APPLY_POOLING_PATCH=1."
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
        "--require-pooling-patch",
        action="store_true",
        help="Verify pooling patch behavior before server startup.",
    )
    parser.add_argument(
        "--require-runtime-compat",
        action="store_true",
        help="Verify local torch/vLLM runtime compatibility before startup.",
    )
    args = parser.parse_args()

    if args.require_pooling_patch:
        require_pooling_patch_ready()
    if args.require_runtime_compat:
        require_runtime_compatibility()
    if not args.require_pooling_patch and not args.require_runtime_compat:
        print("[PREFLIGHT] No checks selected.")


if __name__ == "__main__":
    main()
