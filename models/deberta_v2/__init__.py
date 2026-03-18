"""
Custom DeBERTa v2 encoder for vLLM.

Implements DeBERTa v2/v3 with disentangled self-attention using
vLLM-optimized parallel layers. Compatible with HuggingFace
DeBERTa v2/v3 checkpoints (e.g., microsoft/deberta-v3-base).

Used by: DeBERTa v2/v3-based plugins (GLiNER, NER, classification).
"""

from .config import *  # noqa: F401,F403
from .deberta_v2_encoder import *  # noqa: F401,F403
