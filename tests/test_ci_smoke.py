from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_has_expected_plugin_entrypoints() -> None:
    pyproject = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    entrypoints = data["project"]["entry-points"]["vllm.general_plugins"]

    # Keep this assertion broad enough for easy extension while still
    # catching accidental registry regressions.
    assert len(entrypoints) >= 11
    assert "moderncolbert" in entrypoints
    assert "embeddinggemma" in entrypoints


def test_docs_include_patch_and_support_matrix_links() -> None:
    docs_index = (REPO_ROOT / "docs" / "README.md").read_text()
    assert "pooling_patch.md" in docs_index
    assert "support_matrix.md" in docs_index
