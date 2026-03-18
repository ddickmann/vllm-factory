"""ColQwen3 — vLLM Qwen3-VL + ColPali multi-vector pooler.

Image processing uses the slow Qwen2VLImageProcessor for parity with
the HF reference implementation.
"""

from forge.registration import register_plugin

from .config import ColQwen3Config
from .model import Qwen3VLForColPali


def register() -> None:
    register_plugin("colqwen3", ColQwen3Config,
                     "Qwen3VLForColPali", Qwen3VLForColPali,
                     aliases=["ColQwen3"])


register()
__all__ = ["Qwen3VLForColPali", "ColQwen3Config"]
