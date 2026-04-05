"""Capability detection for the installed vLLM environment.

Detection is based on importing documented plugin interfaces — not on
``if version >= x.y.z`` checks.  Version is recorded as metadata only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("vllm_factory.compat")


@dataclass(frozen=True)
class VllmCapabilities:
    version: str | None = None
    has_general_plugin_group: bool = False
    has_io_processor_group: bool = False
    has_io_processor_interface: bool = False
    has_io_processor_response: bool = False
    pooling_accepts_plugin_task: bool = False
    io_processor_cli_arg_supported: bool = False
    detected_entry_points: dict[str, list[str]] = field(default_factory=dict)


def _vllm_version() -> str | None:
    try:
        import vllm

        return getattr(vllm, "__version__", None)
    except ImportError:
        return None


def _has_entry_point_group(group: str) -> tuple[bool, list[str]]:
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group=group)
        names = [ep.name for ep in eps]
        return len(names) > 0, names
    except Exception:
        return False, []


def _can_import(dotted_path: str) -> bool:
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        return False
    module_path, attr_name = parts
    try:
        import importlib

        mod = importlib.import_module(module_path)
        return hasattr(mod, attr_name)
    except Exception:
        return False


def detect() -> VllmCapabilities:
    """Probe the current environment and return a capabilities snapshot."""
    version = _vllm_version()

    has_general, general_names = _has_entry_point_group("vllm.general_plugins")
    has_io, io_names = _has_entry_point_group("vllm.io_processor_plugins")

    has_interface = _can_import("vllm.plugins.io_processors.interface.IOProcessor")
    has_response = _can_import("vllm.entrypoints.pooling.pooling.protocol.IOProcessorResponse")

    pooling_accepts_plugin = has_interface and has_response

    io_cli = False
    if has_interface:
        try:
            from vllm.engine.arg_utils import EngineArgs

            io_cli = hasattr(EngineArgs, "io_processor_plugin") or _can_import(
                "vllm.engine.arg_utils.EngineArgs"
            )
        except Exception:
            pass

    caps = VllmCapabilities(
        version=version,
        has_general_plugin_group=has_general,
        has_io_processor_group=has_io,
        has_io_processor_interface=has_interface,
        has_io_processor_response=has_response,
        pooling_accepts_plugin_task=pooling_accepts_plugin,
        io_processor_cli_arg_supported=io_cli,
        detected_entry_points={
            "vllm.general_plugins": general_names,
            "vllm.io_processor_plugins": io_names,
        },
    )
    logger.debug("Detected vLLM capabilities: %s", caps)
    return caps
