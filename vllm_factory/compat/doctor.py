"""Diagnostic tool -- ``python -m vllm_factory.compat.doctor``.

Prints a human-readable report of the installed environment:
    - vLLM version
    - available plugin entry points
    - detected general / IO-processor plugin groups
    - whether the native pooling path is available
    - attention-mask patch status
"""

from __future__ import annotations


def run_doctor() -> None:
    from vllm_factory.compat.vllm_capabilities import detect

    caps = detect()

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  vLLM Factory -- Environment Doctor")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"  vLLM version:              {caps.version or 'NOT INSTALLED'}")
    lines.append(f"  general_plugins group:     {'YES' if caps.has_general_plugin_group else 'NO'}")
    lines.append(f"  io_processor_plugins group:{'YES' if caps.has_io_processor_group else 'NO'}")
    lines.append(
        f"  IOProcessor interface:     {'YES' if caps.has_io_processor_interface else 'NO'}"
    )
    lines.append(
        f"  IOProcessorResponse:       {'YES' if caps.has_io_processor_response else 'NO'}"
    )
    lines.append(
        f"  pooling accepts plugin:    {'YES' if caps.pooling_accepts_plugin_task else 'NO'}"
    )
    lines.append(
        f"  --io-processor-plugin CLI: {'YES' if caps.io_processor_cli_arg_supported else 'NO'}"
    )
    lines.append("")

    native_ok = caps.has_io_processor_interface and caps.has_io_processor_response
    if native_ok:
        lines.append("  MODE: NATIVE IO PROCESSOR (no patching required)")
    else:
        lines.append("  MODE: UNSUPPORTED (vLLM >= 0.19 required)")
        lines.append("  ACTION REQUIRED: pip install 'vllm>=0.19'")
    lines.append("")

    from vllm_factory.compat.attention_mask_compat import is_attention_mask_patch_active

    am_active = is_attention_mask_patch_active()
    lines.append(f"  Attention-mask patch:       {'ACTIVE' if am_active else 'NOT APPLIED'}")
    lines.append("    (only needed by linker/rerank plugins)")
    lines.append("")

    gp = caps.detected_entry_points.get("vllm.general_plugins", [])
    io = caps.detected_entry_points.get("vllm.io_processor_plugins", [])
    lines.append(f"  Registered general plugins ({len(gp)}):")
    for name in sorted(gp):
        lines.append(f"    - {name}")
    lines.append(f"  Registered IO processors  ({len(io)}):")
    for name in sorted(io):
        lines.append(f"    - {name}")

    lines.append("")
    lines.append("=" * 60)
    print("\n".join(lines))


if __name__ == "__main__":
    run_doctor()
