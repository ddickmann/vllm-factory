"""General-plugin registration — delegates to forge.registration.

This thin wrapper exists so that ``vllm_factory.registry`` is the canonical
import site for registration, while the actual idempotent helpers remain in
``forge.registration`` (which all existing plugin ``__init__.py`` files
already use).
"""

from __future__ import annotations

from typing import Any, Optional


def register(
    model_type: str,
    config_cls: Any,
    architecture_name: str,
    model_cls: Any,
    *,
    aliases: Optional[list[str]] = None,
) -> None:
    """Register a model + config with HuggingFace AutoConfig and vLLM ModelRegistry."""
    from forge.registration import register_plugin
    register_plugin(model_type, config_cls, architecture_name, model_cls, aliases=aliases)
