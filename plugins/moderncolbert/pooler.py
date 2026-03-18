"""Pooler re-export for moderncolbert plugin.

The ColBERT pooler logic (projection + L2 norm) is embedded directly in model.py
since it is a simple nn.Linear → L2 normalize operation.

For standalone use, the full ColBERT pooler with configurable settings
is available in poolers/colbert.py.
"""

from poolers.colbert import ColBERTPooler

__all__ = ["ColBERTPooler"]
