"""IO-processor plugin registration helpers.

Each IO-processor entry point callable returns the fully-qualified class
name as a string.  This module provides a small helper to verify
resolution at import time (used by the doctor).
"""

from __future__ import annotations


def resolve_io_processor(qualname: str) -> type | None:
    """Attempt to resolve *qualname* to an IOProcessor class."""
    try:
        from vllm.utils import resolve_obj_by_qualname
        return resolve_obj_by_qualname(qualname)
    except Exception:
        pass

    parts = qualname.rsplit(".", 1)
    if len(parts) != 2:
        return None
    module_path, cls_name = parts
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name, None)
    except Exception:
        return None
