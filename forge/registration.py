"""
Registration utilities for vLLM model plugins.

Provides idempotent helpers that handle the boilerplate of registering
custom models and configs with vLLM and HuggingFace.

Usage in a plugin's __init__.py:
    from forge.registration import register_plugin
    register_plugin("my_model_type", MyConfig, "MyModelArch", MyModel)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("forge.registration")


def register_plugin(
    model_type: str,
    config_cls: Any,
    architecture_name: str,
    model_cls: Any,
    *,
    aliases: Optional[list[str]] = None,
) -> None:
    """Register a plugin's config and model with HuggingFace + vLLM.

    This is the main entry point — call this in your plugin's __init__.py.

    Args:
        model_type: The model_type string from config.json
            (e.g., "moderncolbert", "gliner_mmbert").
        config_cls: HuggingFace config class to register with AutoConfig.
        architecture_name: Architecture name for vLLM ModelRegistry
            (e.g., "ModernBertModel", "GLiNERLinkerModel").
        model_cls: The model class to register with vLLM.
        aliases: Optional list of additional architecture names to register
            the same model class under.
    """
    register_config(model_type, config_cls)
    register_with_vllm(architecture_name, model_cls)
    for alias in (aliases or []):
        register_with_vllm(alias, model_cls)


def register_with_vllm(architecture_name: str, model_cls: Any) -> bool:
    """Register a model class with vLLM's ModelRegistry (idempotent).

    Args:
        architecture_name: Architecture name as it appears in model config.
        model_cls: The model class to register.

    Returns:
        True if newly registered, False if already registered.
    """
    try:
        from vllm import ModelRegistry
        ModelRegistry.register_model(architecture_name, model_cls)
        logger.debug("Registered '%s' with vLLM", architecture_name)
        return True
    except (ValueError, KeyError):
        return False
    except ImportError:
        logger.warning("vLLM not installed, skipping registration")
        return False


def register_config(model_type: str, config_cls: Any) -> bool:
    """Register a config class with HuggingFace's AutoConfig (idempotent).

    Args:
        model_type: The model_type string from config.json.
        config_cls: The config class to register.

    Returns:
        True if newly registered, False if already registered.
    """
    try:
        from transformers import AutoConfig
        try:
            AutoConfig.register(model_type, config_cls, exist_ok=True)
        except TypeError:
            # Older transformers without exist_ok
            try:
                AutoConfig.register(model_type, config_cls)
            except ValueError:
                return False
        logger.debug("Registered config '%s' with AutoConfig", model_type)
        return True
    except ImportError:
        logger.warning("transformers not installed, skipping config registration")
        return False
