from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PLUGIN_MATRIX = [
    ("embeddinggemma", "unsloth/embeddinggemma-300m"),
    ("mt5_gliner", "knowledgator/gliner-x-large"),
    ("mmbert_gliner", "VAGOsolutions/SauerkrautLM-GLiNER"),
    ("deberta_gliner", "nvidia/gliner-PII"),
    ("deberta_gliner2", "fastino/gliner2-large-v1"),
    ("deberta_gliner_linker", "knowledgator/gliner-linker-large-v1.0"),
    ("modernbert_gliner_rerank", "knowledgator/gliner-linker-rerank-v1.0"),
    ("moderncolbert", "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT"),
    ("colqwen3", "VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1"),
    ("collfm2", "VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1"),
    ("nemotron_colembed", "nvidia/nemotron-colembed-vl-4b-v2"),
    ("lfm2_colbert", "LiquidAI/LFM2-ColBERT-350M"),
]


@dataclass
class PluginValidationResult:
    plugin: str
    model_ref: str
    import_ok: bool
    resolve_ok: bool
    resolved_model: Optional[str]
    tokenizer: Optional[str]
    command: list[str]
    live_probe_status: str
    live_probe_reason: str
    live_probe_elapsed_s: float
    log_path: Optional[str]


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_markdown_report(
    report_path: Path,
    generated_at: str,
    runtime: dict,
    results: list[PluginValidationResult],
) -> None:
    pass_count = sum(1 for r in results if r.live_probe_status == "pass")
    fail_count = len(results) - pass_count
    lines = [
        "# Plugin Validation Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Python: `{runtime['python']}`",
        f"- Platform: `{runtime['platform']}`",
        f"- venv python: `{runtime['python_executable']}`",
        f"- Pass: `{pass_count}`",
        f"- Fail: `{fail_count}`",
        "",
        "| Plugin | Import | Resolve | Live probe | Reason |",
        "|---|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.plugin}` | {'yes' if r.import_ok else 'no'} | "
            f"{'yes' if r.resolve_ok else 'no'} | `{r.live_probe_status}` | "
            f"{r.live_probe_reason.replace('|', '/')} |"
        )
    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")
    for r in results:
        lines.extend(
            [
                f"### `{r.plugin}`",
                f"- Model ref: `{r.model_ref}`",
                f"- Resolved model: `{r.resolved_model}`",
                f"- Tokenizer: `{r.tokenizer}`",
                f"- Command: `{' '.join(r.command)}`",
                f"- Live probe: `{r.live_probe_status}` ({r.live_probe_reason})",
                f"- Elapsed: `{r.live_probe_elapsed_s:.1f}s`",
                f"- Log: `{r.log_path}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines))


def _classify_probe_output(exit_code: Optional[int], output: str) -> tuple[str, str]:
    text = output.lower()
    if "uvicorn running" in text or "application startup complete" in text:
        return "pass", "server-started"
    if "probe_status: pass" in text:
        return "pass", "probe-script-success"
    if "probe_timeout" in text:
        return "fail", "probe-timeout"
    if "incompatible torch runtime" in text:
        return "fail", "runtime-incompatible"
    if "pooling patch verification failed" in text:
        return "fail", "pooling-patch-failed"
    if "could not infer gliner plugin" in text:
        return "fail", "gliner-plugin-inference-failed"
    if "probe_import_ok: false" in text:
        return "fail", "import-failed"
    if "probe_resolve_ok: false" in text:
        return "fail", "resolve-failed"
    if "engine core initialization failed" in text:
        return "fail", "engine-init-failed"
    if exit_code == 0:
        return "pass", "probe-script-success"
    return "fail", "serve-process-failed"


def _extract_probe_json(output: str) -> dict:
    marker = "PROBE_JSON:"
    for line in output.splitlines():
        if line.startswith(marker):
            raw = line[len(marker):].strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
    return {}


def _run_live_probe(
    plugin: str,
    model_ref: str,
    port: int,
    max_seconds: int,
    logs_dir: Path,
) -> tuple[PluginValidationResult, str]:
    script = f"""
import importlib
import json
import traceback
from forge.server import ModelServer

payload = {{
    "import_ok": False,
    "resolve_ok": False,
    "resolved_model": None,
    "tokenizer": None,
    "command": [],
}}

try:
    importlib.import_module("plugins.{plugin}")
    payload["import_ok"] = True
except Exception as e:
    print("PROBE_IMPORT_ERROR:", type(e).__name__, str(e))

server = ModelServer(
    name="validate-{plugin}",
    model="{model_ref}",
    port={port},
    enforce_eager=True,
    startup_timeout={max_seconds},
    health_check_interval=1.0,
    gliner_plugin="{plugin}",
)

try:
    server._resolve_model_for_server()
    payload["resolve_ok"] = True
    payload["resolved_model"] = server.model
    payload["tokenizer"] = server.tokenizer
    payload["command"] = server._build_command()
except Exception as e:
    print("PROBE_RESOLVE_ERROR:", type(e).__name__, str(e))
    traceback.print_exc()

if payload["import_ok"] and payload["resolve_ok"]:
    try:
        server.start()
        print("PROBE_STATUS: PASS")
    except Exception as e:
        print("PROBE_STATUS: FAIL")
        print("PROBE_ERROR_TYPE:", type(e).__name__)
        print("PROBE_ERROR:", str(e))
        traceback.print_exc()
    finally:
        server.stop()

print("PROBE_JSON:", json.dumps(payload))
"""
    env = os.environ.copy()
    python_bin_dir = os.path.dirname(sys.executable)
    env["PATH"] = f"{python_bin_dir}:{env.get('PATH', '')}"
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=max_seconds + 30,
        )
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        exit_code: Optional[int] = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        output = stdout + ("\n" + stderr if stderr else "") + "\nPROBE_TIMEOUT"
        exit_code = None
    elapsed = time.perf_counter() - started
    status, reason = _classify_probe_output(exit_code, output)
    probe_json = _extract_probe_json(output)
    log_path = logs_dir / f"{plugin}.log"
    log_path.write_text(output)

    result = PluginValidationResult(
        plugin=plugin,
        model_ref=model_ref,
        import_ok=bool(probe_json.get("import_ok", False)),
        resolve_ok=bool(probe_json.get("resolve_ok", False)),
        resolved_model=probe_json.get("resolved_model"),
        tokenizer=probe_json.get("tokenizer"),
        command=probe_json.get("command", []),
        live_probe_status=status,
        live_probe_reason=reason,
        live_probe_elapsed_s=elapsed,
        log_path=str(log_path),
    )
    return result, output


def validate_all(
    max_seconds: int,
    start_port: int,
    output_dir: Path,
) -> tuple[dict, list[PluginValidationResult]]:
    _ensure_output_dir(output_dir)
    logs_dir = _ensure_output_dir(output_dir / "logs")
    runtime = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
    }
    results: list[PluginValidationResult] = []
    for idx, (plugin, model_ref) in enumerate(PLUGIN_MATRIX):
        result, _ = _run_live_probe(
            plugin=plugin,
            model_ref=model_ref,
            port=start_port + idx,
            max_seconds=max_seconds,
            logs_dir=logs_dir,
        )
        results.append(result)
    return runtime, results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate all 11 vLLM Factory plugins.")
    parser.add_argument("--max-startup-seconds", type=int, default=90)
    parser.add_argument("--start-port", type=int, default=8400)
    parser.add_argument("--output-dir", default="benchmarks/results/plugin_validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    runtime, results = validate_all(
        max_seconds=args.max_startup_seconds,
        start_port=args.start_port,
        output_dir=output_dir,
    )
    generated_at = runtime["generated_at"].replace(":", "-")
    json_path = output_dir / f"report-{generated_at}.json"
    md_path = output_dir / f"report-{generated_at}.md"
    payload = {
        "runtime": runtime,
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    _write_markdown_report(md_path, runtime["generated_at"], runtime, results)

    pass_count = sum(1 for r in results if r.live_probe_status == "pass")
    print(f"[VALIDATE] Pass: {pass_count} / {len(results)}")
    print(f"[VALIDATE] JSON report: {json_path}")
    print(f"[VALIDATE] Markdown report: {md_path}")
    return 0 if pass_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
