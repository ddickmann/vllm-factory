"""ModernColBERT — Custom ModernBERT encoder + ColBERT multi-vector pooler.

Registers 'ModernBertModel' (overrides stock vLLM) and 'ModernBertForColBERT' (explicit).
"""

from forge.registration import register_plugin

from .config import ModernColBERTConfig
from .model import ModernBertForColBERT


def register() -> None:
    register_plugin("moderncolbert", ModernColBERTConfig,
                     "ModernBertModel", ModernBertForColBERT,
                     aliases=["ModernBertForColBERT"])


register()

__all__ = ["ModernColBERTConfig", "ModernBertForColBERT"]
