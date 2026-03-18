"""ColLFM2 — vLLM LFM2-VL + ColPali multi-vector pooler.

Registers 'LFM2VLForColPali' (primary) and 'ColLFM2' (alias used in
VAGOsolutions checkpoint config.json).
"""

from forge.registration import register_plugin

from .config import ColLFM2Config
from .model import LFM2VLForColPali


def register() -> None:
    register_plugin("collfm2", ColLFM2Config,
                     "LFM2VLForColPali", LFM2VLForColPali,
                     aliases=["ColLFM2"])


register()
__all__ = ["LFM2VLForColPali", "ColLFM2Config"]
