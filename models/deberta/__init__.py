"""
Custom DeBERTa encoder for vLLM.

Implements DeBERTa v1 with disentangled self-attention using
vLLM-optimized parallel layers. Compatible with HuggingFace
DeBERTa checkpoints (e.g., microsoft/deberta-base).

Used by: DeBERTa-based plugins (GLiNER, NER, classification).
"""

from .config import *  # noqa: F401,F403
from .deberta_encoder import *  # noqa: F401,F403
