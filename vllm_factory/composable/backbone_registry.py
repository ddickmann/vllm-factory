"""Backbone registry — maps backbone names to model classes and hidden-state extractors.

Only backbones shipped in ``models/`` are registered. Users cannot register
arbitrary backbones without code; this is intentional.

Each backbone entry includes:
- A ``create_instance`` callable that knows the exact constructor signature.
- A ``get_hidden_states`` callable that knows how to extract encoder hidden
  states from the backbone's forward output.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

logger = logging.getLogger(__name__)

_MODELS_ROOT = Path(__file__).resolve().parents[2] / "models"

GetHiddenStatesFn = Callable[..., torch.Tensor]
CreateInstanceFn = Callable[..., nn.Module]


@dataclass(frozen=True)
class BackboneEntry:
    name: str
    module_path: Path
    class_name: str
    get_hidden_states: GetHiddenStatesFn
    create_instance: CreateInstanceFn
    extra_model_types: tuple[str, ...] = field(default_factory=tuple)
    extra_architectures: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, BackboneEntry] = {}


def register_backbone(entry: BackboneEntry) -> None:
    _REGISTRY[entry.name] = entry


def get_backbone(name: str) -> BackboneEntry:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown backbone '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_backbones() -> list[str]:
    return list(_REGISTRY.keys())


def _import_module(path: Path, mod_name: str) -> Any:
    """Import a Python module from an absolute file path."""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_backbone_class(entry: BackboneEntry) -> type[nn.Module]:
    """Dynamically import and return the backbone nn.Module class."""
    mod_name = f"vllm_factory._backbone_{entry.name}"
    mod = _import_module(entry.module_path, mod_name)
    return getattr(mod, entry.class_name)


# ---------------------------------------------------------------------------
# Per-backbone constructors — handles signature differences
# ---------------------------------------------------------------------------


def _create_standard(cls: type, vllm_config: Any, prefix: str) -> nn.Module:
    """ModernBERT, DeBERTa, DeBERTa-v2: (vllm_config=, prefix=)."""
    return cls(vllm_config=vllm_config, prefix=prefix)


def _create_mt5(cls: type, vllm_config: Any, prefix: str) -> nn.Module:
    """MT5Encoder: (cfg, cache_config, quant_config, prefix)."""
    cfg = vllm_config.model_config.hf_config
    cache_config = getattr(vllm_config, "cache_config", None)
    quant_config = getattr(vllm_config, "quant_config", None)
    return cls(cfg, cache_config, quant_config, prefix)


def _create_t5gemma2(cls: type, vllm_config: Any, prefix: str) -> nn.Module:
    """T5Gemma2Encoder: (config_or_vllm_config, *, prefix='encoder')."""
    return cls(vllm_config, prefix=prefix or "encoder")


# ---------------------------------------------------------------------------
# Per-backbone hidden-state extractors
# ---------------------------------------------------------------------------


def _hs_generic_forward(
    model: nn.Module, input_ids, positions=None, **kw
) -> torch.Tensor:
    """For encoder-only models whose forward() returns hidden states directly."""
    if positions is not None and positions.dim() == 1:
        positions = positions.unsqueeze(0)
    out = model(input_ids=input_ids, position_ids=positions, **kw)
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, torch.Tensor):
        return out
    raise TypeError(f"Unexpected backbone output type: {type(out)}")


def _hs_deberta(model: nn.Module, input_ids, positions=None, **kw) -> torch.Tensor:
    """DeBERTa v1/v2 returns a plain tensor from forward()."""
    if positions is not None and positions.dim() == 1:
        positions = positions.unsqueeze(0)
    out = model(input_ids=input_ids, position_ids=positions, **kw)
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out


def _hs_mt5(model: nn.Module, input_ids, positions=None, **kw) -> torch.Tensor:
    attention_mask = kw.pop("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return model(input_ids=input_ids, attention_mask=attention_mask, **kw)


def _hs_t5gemma2(model: nn.Module, input_ids, positions=None, **kw) -> torch.Tensor:
    attention_mask = kw.pop("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=positions,
        **kw,
    )


# ---------------------------------------------------------------------------
# Built-in backbone registrations
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    _builtins = [
        BackboneEntry(
            name="modernbert",
            module_path=_MODELS_ROOT / "modernbert" / "modernbert_encoder.py",
            class_name="ModernBertModel",
            get_hidden_states=_hs_generic_forward,
            create_instance=_create_standard,
            extra_model_types=(),
            extra_architectures=(
                "ModernBertForMaskedLM",
                "ModernBertForSequenceClassification",
                "ModernBertForTokenClassification",
            ),
        ),
        BackboneEntry(
            name="deberta",
            module_path=_MODELS_ROOT / "deberta" / "deberta_encoder.py",
            class_name="DebertaEncoderModel",
            get_hidden_states=_hs_deberta,
            create_instance=_create_standard,
            extra_model_types=(),
            extra_architectures=(
                "DebertaModel",
                "DebertaForMaskedLM",
                "DebertaForSequenceClassification",
            ),
        ),
        BackboneEntry(
            name="deberta_v2",
            module_path=_MODELS_ROOT / "deberta_v2" / "deberta_v2_encoder.py",
            class_name="DebertaV2EncoderModel",
            get_hidden_states=_hs_deberta,
            create_instance=_create_standard,
            extra_model_types=("deberta-v2", "deberta_v2"),
            extra_architectures=(
                "DebertaV2Model",
                "DebertaV2ForMaskedLM",
                "DebertaV2ForSequenceClassification",
            ),
        ),
        BackboneEntry(
            name="mt5",
            module_path=_MODELS_ROOT / "mt5" / "mt5_encoder.py",
            class_name="MT5Encoder",
            get_hidden_states=_hs_mt5,
            create_instance=_create_mt5,
            extra_model_types=(),
            extra_architectures=(
                "MT5ForConditionalGeneration",
                "MT5EncoderModel",
            ),
        ),
        BackboneEntry(
            name="t5gemma2",
            module_path=_MODELS_ROOT / "t5gemma2" / "t5gemma2_encoder.py",
            class_name="T5Gemma2Encoder",
            get_hidden_states=_hs_t5gemma2,
            create_instance=_create_t5gemma2,
            extra_model_types=(),
            extra_architectures=(
                "T5Gemma2ForConditionalGeneration",
            ),
        ),
    ]
    for entry in _builtins:
        register_backbone(entry)


_register_builtins()
