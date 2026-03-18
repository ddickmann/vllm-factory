"""Pooler re-export for mmbert_gliner plugin.

Uses the shared GLiNER span pooler from poolers/gliner.py.
"""

from poolers.gliner import GLiNERSpanPooler

__all__ = ["GLiNERSpanPooler"]
