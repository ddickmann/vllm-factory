"""Composable pooling plugin — backbone + pooler composition via registries.

Entry-point for ``vllm.general_plugins``::

    composable_pooling = "vllm_factory.composable:register"

Calling ``register()`` makes the ``ComposablePoolingModel`` architecture
available to vLLM so that ``VLLM_FACTORY_POOLER=<name>`` works at serve time.

This module must NOT call ``register()`` at import time — the entry-point
mechanism in pyproject.toml already calls it.  A module-level call would
trigger vLLM imports too early and cause double-registration warnings.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def register() -> None:
    """Register ComposablePoolingModel with vLLM's ModelRegistry."""
    from forge.registration import register_with_vllm
    from vllm_factory.composable.model import ComposablePoolingModel

    register_with_vllm("ComposablePoolingModel", ComposablePoolingModel)

    pooler = os.environ.get("VLLM_FACTORY_POOLER", "")
    if pooler:
        logger.info("[composable] Pooler override: %s", pooler)
