"""Compatibility layer — the only place that knows which vLLM version is installed.

Public surface:
    VllmCapabilities  — detected capabilities dataclass
    detect()          — build a VllmCapabilities for the current environment
    PoolingBridge     — protocol for native / legacy transport bridges
"""

from vllm_factory.compat.vllm_capabilities import VllmCapabilities, detect

__all__ = ["VllmCapabilities", "detect"]
