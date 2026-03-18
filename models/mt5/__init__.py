"""
Custom mT5 encoder-decoder for vLLM.

Implements T5-style attention with relative position bias (RPB)
using optimized Triton kernels, plus fused feed-forward networks.

Used by: mT5-GLiNER, mT5-Span Predictor plugins.
"""

from .config import *  # noqa: F401,F403
from .mt5_encoder import *  # noqa: F401,F403
