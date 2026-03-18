#!/usr/bin/env python3
"""
Patch vLLM 0.15.x /pooling HTTP endpoint for custom pooling models.

Two patches applied:

1. REQUEST SIDE -- extra_kwargs passthrough:
   vLLM's PoolingParams has an `extra_kwargs` field that custom poolers
   (GLiNER, span predictor, ColBERT, etc.) use to receive structured metadata.
   However, the HTTP /pooling endpoint's protocol classes do NOT expose this
   field, so metadata sent via HTTP is silently dropped.

   Fix: Add `extra_kwargs: dict | None` to PoolingCompletionRequest and
   PoolingChatRequest, and pass it through in to_pooling_params().

2. RESPONSE SIDE -- PoolingResponseData.data type:
   Custom pooling models (GLiNER, span predictor) return multi-dimensional
   tensors (e.g. 3D logits of shape [L, max_width, num_classes]). vLLM's
   PoolingResponseData validates `data` as `list[list[float]] | list[float] | str`
   which rejects anything deeper than 2D, causing pydantic validation errors.

   Fix: Change `data` field type to `Any` to accept arbitrary tensor shapes.

Both patches are idempotent -- safe to run multiple times.

NOTE: This patch is specific to vLLM 0.15.x. Future vLLM versions may
incorporate these fixes upstream, or the protocol file location may change.
Always re-verify after upgrading vLLM.

Usage:
    # In Dockerfile (before starting the server):
    RUN python -m forge.patches.pooling_extra_kwargs

    # Or from Python:
    from forge.patches.pooling_extra_kwargs import apply_patch
    apply_patch()
"""

import os
import re
import site
import sys
from importlib import metadata

from packaging.version import InvalidVersion, Version

SUPPORTED_VLLM_MIN = Version("0.15.0")
SUPPORTED_VLLM_MAX_EXCLUSIVE = Version("0.16.0")


def find_protocol_file() -> str:
    """Locate the vLLM pooling protocol file."""
    for prefix in site.getsitepackages() + [site.getusersitepackages()]:
        candidate = os.path.join(
            prefix,
            "vllm", "entrypoints", "pooling", "pooling", "protocol.py",
        )
        if os.path.isfile(candidate):
            return candidate

    # Fallback: search sys.path
    for p in sys.path:
        candidate = os.path.join(
            p, "vllm", "entrypoints", "pooling", "pooling", "protocol.py"
        )
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Cannot locate vllm/entrypoints/pooling/pooling/protocol.py. "
        "Is vLLM 0.15.x installed?"
    )


def get_installed_vllm_version() -> str | None:
    """Return installed vLLM version string, if available."""
    try:
        return metadata.version("vllm")
    except metadata.PackageNotFoundError:
        return None


def ensure_supported_vllm_version(strict: bool = True) -> bool:
    """Validate that installed vLLM is in the supported patch range.

    Set VLLM_FACTORY_ALLOW_UNSUPPORTED_VLLM=1 to bypass strict mode.
    """
    raw = get_installed_vllm_version()
    if raw is None:
        msg = (
            "[PATCH] vLLM is not installed in this environment. "
            "Install vLLM first, then re-run the patch."
        )
        if strict:
            raise RuntimeError(msg)
        print(msg)
        return False

    try:
        parsed = Version(raw)
    except InvalidVersion:
        msg = (
            f"[PATCH] Unable to parse vLLM version '{raw}'. "
            "Proceeding may fail because this patch targets vLLM 0.15.x."
        )
        if strict and os.getenv("VLLM_FACTORY_ALLOW_UNSUPPORTED_VLLM") != "1":
            raise RuntimeError(msg)
        print(msg)
        return False

    supported = SUPPORTED_VLLM_MIN <= parsed < SUPPORTED_VLLM_MAX_EXCLUSIVE
    if supported:
        print(f"[PATCH] Detected supported vLLM version: {parsed}")
        return True

    msg = (
        f"[PATCH] Unsupported vLLM version: {parsed}. "
        f"This patch supports >= {SUPPORTED_VLLM_MIN} and < {SUPPORTED_VLLM_MAX_EXCLUSIVE}. "
        "Set VLLM_FACTORY_ALLOW_UNSUPPORTED_VLLM=1 to bypass at your own risk."
    )
    if strict and os.getenv("VLLM_FACTORY_ALLOW_UNSUPPORTED_VLLM") != "1":
        raise RuntimeError(msg)
    print(msg)
    return False


def _ensure_any_import(content: str) -> str:
    """Ensure 'Any' is in the typing import."""
    if re.search(r"from typing import .*?\bAny\b", content):
        return content

    match = re.search(r"^from typing import (?P<items>.+)$", content, flags=re.MULTILINE)
    if match:
        raw_items = match.group("items")
        items = [item.strip() for item in raw_items.split(",") if item.strip()]
        if "Any" not in items:
            items.insert(0, "Any")
        new_line = f"from typing import {', '.join(items)}"
        return content[:match.start()] + new_line + content[match.end():]

    insertion = "from typing import Any\n"
    future_match = re.search(r"^from __future__ import .+$", content, flags=re.MULTILINE)
    if future_match:
        insert_at = future_match.end() + 1
        return content[:insert_at] + insertion + content[insert_at:]

    return insertion + content


def _find_class_span(content: str, class_name: str) -> tuple[int, int] | None:
    class_starts = list(re.finditer(r"^class\s+(\w+)\b", content, flags=re.MULTILINE))
    target_idx = None
    for idx, match in enumerate(class_starts):
        if match.group(1) == class_name:
            target_idx = idx
            break

    if target_idx is None:
        return None

    start = class_starts[target_idx].start()
    if target_idx + 1 < len(class_starts):
        end = class_starts[target_idx + 1].start()
    else:
        end = len(content)
    return (start, end)


def _replace_class_block(content: str, class_name: str, new_block: str) -> str:
    span = _find_class_span(content, class_name)
    if span is None:
        return content

    start, end = span
    return content[:start] + new_block.rstrip() + "\n\n" + content[end:]


def _inject_extra_kwargs_field_in_class(
    class_block: str,
    description: str,
) -> tuple[str, bool]:
    if "extra_kwargs:" in class_block:
        return class_block, False

    fn_match = re.search(r"^(\s*)def to_pooling_params\(self\):", class_block, flags=re.MULTILINE)
    if not fn_match:
        return class_block, False

    indent = fn_match.group(1)
    field = (
        f'{indent}extra_kwargs: dict[str, Any] | None = Field(\n'
        f"{indent}    default=None,\n"
        f'{indent}    description="{description}",\n'
        f"{indent})\n\n"
    )
    insert_at = fn_match.start()
    return class_block[:insert_at] + field + class_block[insert_at:], True


def _inject_extra_kwargs_in_pooling_params(class_block: str) -> tuple[str, bool]:
    if "extra_kwargs=self.extra_kwargs" in class_block:
        return class_block, False

    method_pattern = (
        r"(def to_pooling_params\(self\):\n"
        r"\s+return PoolingParams\(\n)"
        r"(?P<args>[\s\S]*?)"
        r"(?P<close>\n\s+\))"
    )
    match = re.search(method_pattern, class_block)
    if not match:
        return class_block, False

    args = match.group("args")
    arg_indent_match = re.search(r"^(\s*)\w", args, flags=re.MULTILINE)
    arg_indent = arg_indent_match.group(1) if arg_indent_match else "        "
    injected = args + f"{arg_indent}extra_kwargs=self.extra_kwargs,\n"

    new_block = (
        class_block[:match.start("args")]
        + injected
        + class_block[match.end("args"):]
    )
    return new_block, True


def _patch_request_class(
    content: str,
    class_name: str,
    field_description: str,
) -> tuple[str, bool]:
    span = _find_class_span(content, class_name)
    if span is None:
        return content, False

    start, end = span
    class_block = content[start:end]
    class_block_new, changed_field = _inject_extra_kwargs_field_in_class(
        class_block, field_description
    )
    class_block_new, changed_call = _inject_extra_kwargs_in_pooling_params(class_block_new)
    changed = changed_field or changed_call
    if not changed:
        return content, False

    return _replace_class_block(content, class_name, class_block_new), True


def _patch_extra_kwargs(content: str) -> tuple[str, bool]:
    """Patch request classes to pass extra_kwargs through to PoolingParams.

    Returns (new_content, changed).
    """
    if (
        "extra_kwargs=self.extra_kwargs" in content
        and "extra_kwargs: dict[str, Any] | None" in content
    ):
        print("[PATCH] extra_kwargs: already patched")
        return content, False

    content, changed_completion = _patch_request_class(
        content,
        "PoolingCompletionRequest",
        "Extra keyword arguments passed through to the custom pooler via "
        "PoolingParams.extra_kwargs. Used by custom pooling models (GLiNER, "
        "span predictor, ColBERT, etc.) to receive structured metadata such "
        "as entity_spans, words_mask, etc.",
    )
    if changed_completion:
        print("[PATCH] extra_kwargs: PoolingCompletionRequest patched")
    else:
        print("[PATCH] WARNING: PoolingCompletionRequest pattern not found")
    content, changed_chat = _patch_request_class(
        content,
        "PoolingChatRequest",
        "Extra keyword arguments passed through to the custom pooler via "
        "PoolingParams.extra_kwargs.",
    )
    if changed_chat:
        print("[PATCH] extra_kwargs: PoolingChatRequest patched")
    else:
        print("[PATCH] WARNING: PoolingChatRequest pattern not found")

    return content, (changed_completion or changed_chat)


def _patch_response_data(content: str) -> tuple[str, bool]:
    """Patch PoolingResponseData.data to accept Any (3D+ tensors).

    Custom pooling models like GLiNER return multi-dimensional logits
    (e.g. shape [L, max_width, num_classes]) which are serialized as
    nested lists. vLLM's default type rejects anything deeper than 2D.

    Returns (new_content, changed).
    """
    if re.search(r'data:\s*Any', content):
        print("[PATCH] PoolingResponseData.data: already patched")
        return content, False

    span = _find_class_span(content, "PoolingResponseData")
    if span:
        start, end = span
        class_block = content[start:end]
        field_match = re.search(r"^(\s*)data:\s*.+$", class_block, flags=re.MULTILINE)
        if not field_match:
            print("[PATCH] WARNING: PoolingResponseData.data field not found")
            return content, False

        indent = field_match.group(1)
        new_field = (
            f"{indent}data: Any  # Accept arbitrary nesting depth (1D, 2D, 3D tensors, "
            "or base64 str) -- required for GLiNER/span-predictor logits"
        )
        class_block_new = (
            class_block[:field_match.start()]
            + new_field
            + class_block[field_match.end():]
        )
        content = _replace_class_block(content, "PoolingResponseData", class_block_new)
        print("[PATCH] PoolingResponseData.data: patched to accept Any")
        return content, True

    print("[PATCH] WARNING: PoolingResponseData.data pattern not found")
    return content, False


def patch_file(path: str) -> bool:
    """Apply all patches to the protocol file. Returns True if changes were made."""
    with open(path, "r") as f:
        content = f.read()

    content = _ensure_any_import(content)
    content, changed1 = _patch_extra_kwargs(content)
    content, changed2 = _patch_response_data(content)

    if not changed1 and not changed2:
        print(f"[PATCH] No changes needed: {path}")
        return False

    with open(path, "w") as f:
        f.write(content)

    # Remove cached bytecode
    cache_dir = os.path.join(os.path.dirname(path), "__pycache__")
    if os.path.isdir(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)

    print(f"[PATCH] Successfully patched: {path}")
    return True


def verify_patch() -> bool:
    """Verify both patches work by importing and testing."""
    # Force reimport
    mods_to_remove = [
        k for k in sys.modules
        if "vllm.entrypoints.pooling" in k
    ]
    for k in mods_to_remove:
        del sys.modules[k]

    ok = True

    # 1. Verify extra_kwargs passthrough
    try:
        from vllm.entrypoints.pooling.pooling.protocol import PoolingCompletionRequest
        req = PoolingCompletionRequest(
            input="test",
            extra_kwargs={"entity_spans": [[0, 1]], "test": True},
        )
        pp = req.to_pooling_params()
        assert pp.extra_kwargs is not None, "extra_kwargs not passed through"
        assert pp.extra_kwargs["test"] is True, "extra_kwargs data corrupted"
        print("[PATCH] Verify: extra_kwargs passthrough OK")
    except Exception as e:
        print(f"[PATCH] Verify FAILED (extra_kwargs): {e}")
        ok = False

    # 2. Verify PoolingResponseData accepts 3D data
    try:
        from vllm.entrypoints.pooling.pooling.protocol import PoolingResponseData
        resp1d = PoolingResponseData(index=0, data=[0.1, 0.2, 0.3])
        assert resp1d.data == [0.1, 0.2, 0.3], "1D data failed"

        resp2d = PoolingResponseData(index=0, data=[[0.1, 0.2], [0.3, 0.4]])
        assert len(resp2d.data) == 2, "2D data failed"

        data_3d = [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
        resp3d = PoolingResponseData(index=0, data=data_3d)
        assert len(resp3d.data) == 2, "3D data failed"

        print("[PATCH] Verify: PoolingResponseData accepts 1D/2D/3D data OK")
    except Exception as e:
        print(f"[PATCH] Verify FAILED (PoolingResponseData): {e}")
        ok = False

    return ok


def apply_patch() -> bool:
    """Apply the pooling protocol patches. Returns True if successful."""
    ensure_supported_vllm_version(strict=True)
    path = find_protocol_file()
    print(f"[PATCH] Target: {path}")
    patch_file(path)
    return verify_patch()


if __name__ == "__main__":
    ensure_supported_vllm_version(strict=True)
    path = find_protocol_file()
    print(f"[PATCH] Target: {path}")

    changed = patch_file(path)
    ok = verify_patch()

    if not ok:
        sys.exit(1)

    print("[PATCH] Done.")
