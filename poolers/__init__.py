"""
Pooler heads — reusable pooling layers for vLLM custom models.

Each pooler implements the FactoryPooler protocol (zero vLLM imports)
and is wrapped by VllmPoolerAdapter at model construction time.

Available poolers:
    - ColBERTPooler: Token-level multi-vector embeddings
    - GLiNERSpanPooler: Span extraction for zero-shot NER
    - ColPaliPooler: Multi-vector vision-language embeddings
    - GLiNER2Pooler: Schema-based multi-task extraction
"""

from poolers.colbert import ColBERTPooler
from poolers.colpali import ColPaliPooler
from poolers.gliner import GLiNERSpanPooler
from poolers.gliner2 import GLiNER2Pooler

__all__ = ["ColBERTPooler", "GLiNERSpanPooler", "ColPaliPooler", "GLiNER2Pooler"]
