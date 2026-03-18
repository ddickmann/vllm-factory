"""
Pooler heads — reusable pooling layers for vLLM custom models.

Each pooler is a standalone nn.Module that transforms encoder hidden states
into task-specific outputs (embeddings, logits, classifications).

Available poolers:
    - ColBERTPooler: Token-level multi-vector embeddings
    - GLiNERSpanPooler: Span extraction for zero-shot NER
    - ColPaliPooler: Multi-vector vision-language embeddings
"""

from poolers.colbert import ColBERTPooler
from poolers.colpali import ColPaliPooler
from poolers.gliner import GLiNERSpanPooler

__all__ = ["ColBERTPooler", "GLiNERSpanPooler", "ColPaliPooler"]
