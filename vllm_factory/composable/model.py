"""ComposablePoolingModel — generic backbone + pooler composition.

A single ``nn.Module`` that dynamically wires *any* registered backbone
with *any* registered pooler at construction time, based on:

1. ``VLLM_FACTORY_POOLER`` env var  (set via ``--override-pooler`` or directly)
2. ``pooler_type`` field in the checkpoint's ``config.json``

No new model files needed per combination.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig

from vllm_factory.composable.backbone_registry import (
    get_backbone,
    list_backbones,
    load_backbone_class,
)
from vllm_factory.composable.pooler_registry import get_pooler_cls, list_poolers
from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

logger = logging.getLogger(__name__)


def _build_resolution_maps() -> (
    tuple[dict[str, str], dict[str, str]]
):
    """Build model_type->backbone and architecture->backbone maps from the registry."""
    type_map: dict[str, str] = {}
    arch_map: dict[str, str] = {}

    from vllm_factory.composable.backbone_registry import _REGISTRY

    for name, entry in _REGISTRY.items():
        type_map[name] = name
        for extra_mt in entry.extra_model_types:
            type_map[extra_mt] = name
        arch_map[entry.class_name] = name
        for extra_arch in entry.extra_architectures:
            arch_map[extra_arch] = name

    return type_map, arch_map


def _resolve_backbone_name(hf_config: Any) -> str:
    """Determine which backbone to use from the HF config."""
    type_map, arch_map = _build_resolution_maps()

    model_type = getattr(hf_config, "model_type", "")

    if model_type in type_map:
        return type_map[model_type]

    architectures = getattr(hf_config, "architectures", []) or []
    for arch in architectures:
        if arch in arch_map:
            return arch_map[arch]

    raise ValueError(
        f"Cannot determine backbone for model_type='{model_type}', "
        f"architectures={architectures}. "
        f"Available backbones: {list_backbones()}"
    )


def _resolve_pooler_name(hf_config: Any) -> str:
    """Determine which pooler to use: env > config > error."""
    override = os.environ.get("VLLM_FACTORY_POOLER", "").strip()
    if override:
        return override

    pooler_type = getattr(hf_config, "pooler_type", None)
    if pooler_type:
        return pooler_type

    raise ValueError(
        "No pooler specified. Set VLLM_FACTORY_POOLER=<name> or add "
        f"pooler_type to config.json. Available: {list_poolers()}"
    )


def _instantiate_pooler(pooler_cls: type, hidden_size: int, pooler_config: dict) -> Any:
    """Try to instantiate a pooler with best-effort argument adaptation."""
    # Standard path: (hidden_size, **config)
    try:
        return pooler_cls(hidden_size=hidden_size, **pooler_config)
    except TypeError:
        pass

    # GLiNER-style: (cfg) where cfg is a namespace with hidden_size attribute
    try:
        return pooler_cls(hidden_size, **pooler_config)
    except TypeError:
        pass

    # No-args fallback
    try:
        return pooler_cls()
    except TypeError as exc:
        raise TypeError(
            f"Cannot instantiate pooler {pooler_cls.__name__}. "
            "It must accept either (hidden_size=, **kwargs), (hidden_size), "
            "or no arguments."
        ) from exc


class ComposablePoolingModel(nn.Module):
    """Generic backbone + pooler model — zero custom code per combination.

    Construction:
        1. Resolves backbone name from HF config model_type / architectures
        2. Resolves pooler name from env var / config
        3. Instantiates backbone via per-backbone ``create_instance``
        4. Instantiates pooler (stateless or nn.Module)
        5. Wraps pooler in VllmPoolerAdapter
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config

        backbone_name = _resolve_backbone_name(hf_config)
        pooler_name = _resolve_pooler_name(hf_config)

        logger.info(
            "[ComposablePoolingModel] backbone=%s  pooler=%s",
            backbone_name,
            pooler_name,
        )

        # --- backbone ---
        backbone_entry = get_backbone(backbone_name)
        BackboneClass = load_backbone_class(backbone_entry)
        self._backbone = backbone_entry.create_instance(
            BackboneClass, vllm_config, prefix
        )
        self._backbone_entry = backbone_entry

        # --- pooler ---
        pooler_cls = get_pooler_cls(pooler_name)
        hidden_size = getattr(hf_config, "hidden_size", 768)
        pooler_config = getattr(hf_config, "pooler_config", None) or {}

        business_pooler = _instantiate_pooler(pooler_cls, hidden_size, pooler_config)

        # Store nn.Module poolers as a proper submodule so their parameters
        # are tracked, moved to the right device/dtype, and appear in
        # state_dict.  Non-module poolers are stored as a plain attribute.
        if isinstance(business_pooler, nn.Module):
            self.business_pooler = business_pooler
        else:
            self._business_pooler_plain = business_pooler

        self.pooler = VllmPoolerAdapter(business_pooler)

    @property
    def _business_pooler(self) -> Any:
        if hasattr(self, "business_pooler"):
            return self.business_pooler
        return self._business_pooler_plain

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run backbone and return hidden states for the pooler."""
        hs = self._backbone_entry.get_hidden_states(
            self._backbone,
            input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if hs.dim() == 3 and hs.shape[0] == 1:
            hs = hs.squeeze(0)
        return hs

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Delegate to backbone's embedding layer."""
        if hasattr(self._backbone, "embed_input_ids"):
            return self._backbone.embed_input_ids(input_ids)
        if hasattr(self._backbone, "get_input_embeddings"):
            return self._backbone.get_input_embeddings(input_ids)
        for attr_name in ("embeddings", "embed_tokens", "word_embeddings"):
            emb = getattr(self._backbone, attr_name, None)
            if emb is not None and callable(emb):
                return emb(input_ids)
        raise AttributeError(
            f"Backbone {type(self._backbone).__name__} has no embed_input_ids "
            "or known embedding attribute."
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str] | None:
        """Delegate weight loading to backbone, then load pooler weights if any."""
        backbone_weights = []
        pooler_weights: dict[str, torch.Tensor] = {}
        pooler_prefix = "pooler_head."

        for name, tensor in weights:
            if name.startswith(pooler_prefix):
                pooler_weights[name[len(pooler_prefix):]] = tensor
            else:
                backbone_weights.append((name, tensor))

        # Backbone load_weights — not all backbones implement this method.
        # e.g. MT5Encoder lacks it; fall back to manual parameter assignment.
        result: set[str] | None = None
        if hasattr(self._backbone, "load_weights"):
            result = self._backbone.load_weights(backbone_weights)
        else:
            result = self._load_weights_manual(backbone_weights)

        if pooler_weights and isinstance(self._business_pooler, nn.Module):
            self._business_pooler.load_state_dict(pooler_weights, strict=False)
            try:
                device = next(self._backbone.parameters()).device
                dtype = next(self._backbone.parameters()).dtype
                self._business_pooler.to(device=device, dtype=dtype)
            except StopIteration:
                pass
            logger.info(
                "[ComposablePoolingModel] Loaded %d pooler weight tensors",
                len(pooler_weights),
            )

        return result

    def _load_weights_manual(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Fallback weight loader for backbones without load_weights()."""
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        params = dict(self._backbone.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name in params:
                default_weight_loader(params[name], tensor)
                loaded.add(name)
            else:
                logger.debug("Skipping unmatched weight: %s", name)
        return loaded
