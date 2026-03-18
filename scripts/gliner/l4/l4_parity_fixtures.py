"""Shared constants for L4 rerank parity scripts and tests (avoid import cycles)."""

from __future__ import annotations

from typing import List

HF_MODEL_ID = "knowledgator/gliner-linker-rerank-v1.0"

TEST_TEXT = (
    "Farnese Palace is one of the most important palaces in the city of Rome. "
    "Michelangelo also contributed."
)

TEST_LABELS: List[str] = [
    "Farnese Palace: Renaissance palace",
    "Rome: Capital city",
    "Michelangelo: Italian artist",
]

MULTI_TEXTS: List[str] = [
    "Hi.",
    TEST_TEXT,
    "The quick brown fox jumps over the lazy dog. " * 2,
    (
        "Farnese Palace is one of the most important palaces in the city of Rome. "
        "It was designed in 1517. Michelangelo contributed. Today it is an embassy."
    ),
]
