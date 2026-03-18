from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def _safe_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _gpu_summary() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = proc.stdout.strip()
        return out if out else "unknown-or-not-available"
    except FileNotFoundError:
        return "nvidia-smi-not-found"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _safe_version("torch"),
        "vllm": _safe_version("vllm"),
        "gpu": _gpu_summary(),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote environment metadata to {out}")


if __name__ == "__main__":
    main()
