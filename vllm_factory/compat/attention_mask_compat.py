"""Attention-mask forwarding compat — quarantined from generic registration.

The GLiNER linker/rerank plugin family requires a monkey-patch to
``GPUModelRunner._preprocess`` so that ``attention_mask`` from
``PoolingParams.extra_kwargs`` reaches ``model.forward()``.

This is orthogonal to the pooling protocol patch and applies only to
plugins whose collator produces explicit attention masks.

Consumers:
    - deberta_gliner_linker
    - modernbert_gliner_rerank

This module exists so that the patch is:
    1. Applied explicitly (not as a hidden side-effect of import-time registration)
    2. Isolated from the generic plugin registration path
    3. Diagnosable via ``vllm_factory.compat.doctor``
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vllm_factory.compat.attention_mask")

_applied = False


def ensure_attention_mask_patch() -> bool:
    """Apply the GPUModelRunner._preprocess patch if not already applied.

    Returns True if the patch is active (applied now or previously).
    """
    global _applied
    if _applied:
        return True

    try:
        from plugins.deberta_gliner_linker.vllm_pooling_attention_mask import (
            apply_pooling_attention_mask_patch,
        )
        result = apply_pooling_attention_mask_patch()
        _applied = result
        if result:
            logger.info(
                "Attention-mask compat patch applied for linker/rerank plugins"
            )
        return result
    except ImportError:
        logger.warning(
            "vllm_pooling_attention_mask module not found — "
            "attention mask forwarding will not be available"
        )
        return False


def is_attention_mask_patch_active() -> bool:
    """Check whether the attention mask patch is currently active."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        return getattr(GPUModelRunner, "_gliner_linker_preprocess_patched", False)
    except ImportError:
        return False
