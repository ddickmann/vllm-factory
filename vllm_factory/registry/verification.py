"""Entry-point verification — used by the doctor and CI smoke tests."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any


def verify_entry_points() -> dict[str, Any]:
    """Return a summary of discovered vllm-factory entry points."""
    result: dict[str, Any] = {}
    for group in ("vllm.general_plugins", "vllm.io_processor_plugins"):
        eps = list(entry_points(group=group))
        result[group] = {
            "count": len(eps),
            "names": [ep.name for ep in eps],
        }
    return result
