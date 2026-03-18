#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-reference"

echo "=== Creating reference virtual environment ==="
echo "  Location: ${VENV_DIR}"

if [ -d "${VENV_DIR}" ]; then
    echo "  Removing existing venv..."
    rm -rf "${VENV_DIR}"
fi

python3.11 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel -q

echo "  Installing reference dependencies..."
pip install -r "${REPO_ROOT}/requirements-reference.txt" 2>&1 | tail -5

echo ""
echo "=== Reference venv ready ==="
echo "  Activate with: source ${VENV_DIR}/bin/activate"
echo "  Packages:"
pip list 2>/dev/null | grep -iE "gliner|pylate|sauerkraut|sentence" || true
