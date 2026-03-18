"""
Custom ModernBERT encoder for vLLM.

This is a vLLM-optimized implementation with parallel layers
(Column/RowParallelLinear) and fused Triton kernels for attention,
MLP, and LayerNorm.

Used by: ColBERT, mmBERT-GLiNER plugins.
"""

from .config import *  # noqa: F401,F403
from .modernbert_encoder import *  # noqa: F401,F403
