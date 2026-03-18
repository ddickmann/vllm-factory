#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_REF="${REPO_ROOT}/.venv-reference"
REF_ROOT="/tmp/vllm-factory-refs"

if [ ! -d "${VENV_REF}" ]; then
    echo "ERROR: Reference venv not found at ${VENV_REF}"
    echo "Run: bash scripts/create_reference_venv.sh"
    exit 1
fi

PYTHON="${VENV_REF}/bin/python"
mkdir -p "${REF_ROOT}"

echo "============================================================"
echo "  Generating reference outputs for all plugins"
echo "  Python: ${PYTHON}"
echo "  Output: ${REF_ROOT}"
echo "============================================================"

ok=0; fail=0

run_ref() {
    local name="$1"; shift
    echo ""
    echo "--- ${name} ---"
    if "$@" 2>&1; then
        ok=$((ok + 1))
        echo "  ✓ ${name} done"
    else
        fail=$((fail + 1))
        echo "  ✗ ${name} FAILED"
    fi
}

# GLiNER-based plugins (Phase 1: --prepare)
run_ref "deberta_gliner" \
    "${PYTHON}" "${REPO_ROOT}/plugins/deberta_gliner/parity_test.py" --prepare

run_ref "mmbert_gliner" \
    "${PYTHON}" "${REPO_ROOT}/plugins/mmbert_gliner/parity_test.py" --prepare

run_ref "mt5_gliner" \
    "${PYTHON}" "${REPO_ROOT}/plugins/mt5_gliner/parity_test.py" --prepare

run_ref "deberta_gliner2" \
    "${PYTHON}" "${REPO_ROOT}/plugins/deberta_gliner2/parity_test.py" --prepare

run_ref "deberta_gliner_linker" \
    "${PYTHON}" "${REPO_ROOT}/scripts/gliner/l3/parity_test.py" --prepare

# ColBERT / ColPali reference generators
run_ref "moderncolbert" \
    "${PYTHON}" "${REPO_ROOT}/plugins/moderncolbert/generate_reference.py" \
    --model VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT \
    --output-dir "${REF_ROOT}/moderncolbert"

run_ref "colqwen3" \
    "${PYTHON}" "${REPO_ROOT}/plugins/colqwen3/generate_reference.py" \
    --model VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1 \
    --output-dir "${REF_ROOT}/colqwen3"

run_ref "collfm2" \
    "${PYTHON}" "${REPO_ROOT}/plugins/collfm2/generate_reference.py" \
    --model VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1 \
    --output-dir "${REF_ROOT}/collfm2"

# EmbeddingGemma (uses sentence-transformers — also in ref venv)
run_ref "embeddinggemma" \
    "${PYTHON}" "${REPO_ROOT}/plugins/embeddinggemma/parity_test.py" --prepare

echo ""
echo "============================================================"
echo "  Reference generation complete: ✓ ${ok}  ✗ ${fail}"
echo "============================================================"
[ "${fail}" -eq 0 ] || exit 1
