"""
Configuration utilities for vLLM model plugins.

Helpers for extending HuggingFace configs, patching legacy configs,
and safely extracting config attributes.
"""

from __future__ import annotations

from typing import Any


def safe_getattr(config: Any, name: str, default: Any = None) -> Any:
    """Safely get an attribute from a config, returning default if missing."""
    return getattr(config, name, default)


def patch_config_defaults(config: Any, defaults: dict[str, Any]) -> None:
    """Set default values on a config object if they don't exist.

    Args:
        config: HuggingFace config object
        defaults: Dict of {attr_name: default_value}

    Example:
        >>> patch_config_defaults(config, {
        ...     "colbert_dim": 128,
        ...     "query_length": 256,
        ... })
    """
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)


def ensure_config_type(config: Any, target_cls: type, **extra_kwargs: Any) -> Any:
    """Convert a config to a target type if it isn't already.

    Useful when a model's config.json has a base type but you need
    your custom config class.

    Args:
        config: Source config object
        target_cls: Target config class
        **extra_kwargs: Additional kwargs to pass to target_cls constructor

    Returns:
        Config instance of target_cls

    Raises:
        ValueError: If conversion fails
    """
    if isinstance(config, target_cls):
        return config

    try:
        config_dict = config.to_dict()
        config_dict.update(extra_kwargs)
        return target_cls(**config_dict)
    except Exception as e:
        raise ValueError(
            f"Failed to convert config {type(config).__name__} "
            f"to {target_cls.__name__}: {e}"
        ) from e
