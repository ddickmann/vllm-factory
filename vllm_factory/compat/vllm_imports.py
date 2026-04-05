"""Portable vLLM import shims — resolves moved modules across vLLM versions.

Usage:
    from vllm_factory.compat.vllm_imports import Attention, PromptType, PoolingMetadata
"""

from __future__ import annotations

# Attention -----------------------------------------------------------------
try:
    from vllm.attention import Attention
except ImportError:
    from vllm.model_executor.layers.attention import Attention  # noqa: F401

# PromptType ----------------------------------------------------------------
try:
    from vllm.inputs.data import PromptType
except ImportError:
    from vllm.inputs import PromptType  # noqa: F401

# PoolingMetadata / PoolingTensors ------------------------------------------
PoolingTensors = None
try:
    from vllm.model_executor.pooling_metadata import PoolingMetadata, PoolingTensors
except ImportError:
    from vllm.v1.pool.metadata import PoolingMetadata  # noqa: F401

__all__ = ["Attention", "PromptType", "PoolingMetadata", "PoolingTensors"]
