"""
Component registry for vLLM Factory.

Discovers and catalogs available building blocks (models, poolers,
kernels, processors) for the LEGO-style plugin builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Component:
    """A registered building block."""

    name: str
    category: str  # "model", "pooler", "kernel"
    description: str
    module_path: str  # e.g., "models.modernbert_vllm"
    class_name: str | None = None
    requires_custom_encoder: bool = False
    compatible_with: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# ── Built-in component catalog ──────────────────────────────────────

_REGISTRY: dict[str, dict[str, Component]] = {
    "model": {},
    "pooler": {},
    "kernel": {},
}


def register_component(component: Component) -> None:
    """Register a component in the catalog."""
    _REGISTRY[component.category][component.name] = component


def get_component(category: str, name: str) -> Component | None:
    """Look up a component by category and name."""
    return _REGISTRY.get(category, {}).get(name)


def list_components(category: str | None = None) -> list[Component]:
    """List all registered components, optionally filtered by category."""
    if category:
        return list(_REGISTRY.get(category, {}).values())
    return [c for cat in _REGISTRY.values() for c in cat.values()]


# ── Register built-in components ─────────────────────────────────────

def _register_builtins() -> None:
    """Register all built-in building blocks."""

    # ── Base Models ──
    register_component(Component(
        name="modernbert",
        category="model",
        description="ModernBERT encoder with vLLM parallel layers + fused Triton kernels",
        module_path="models.modernbert",
        class_name="ModernBertModel",
        requires_custom_encoder=True,
        tags=["encoder-only", "bidirectional", "triton"],
    ))
    register_component(Component(
        name="mt5",
        category="model",
        description="mT5 encoder-decoder with Flash Attention RPB kernels",
        module_path="models.mt5",
        class_name="MT5EncoderModel",
        requires_custom_encoder=True,
        tags=["encoder-decoder", "multilingual", "triton"],
    ))
    register_component(Component(
        name="vllm:ModernBertModel",
        category="model",
        description="vLLM's built-in ModernBERT (use when you only need a custom pooler)",
        module_path="vllm.model_executor.models.modernbert",
        class_name="ModernBertModel",
        requires_custom_encoder=False,
        tags=["encoder-only", "built-in"],
    ))
    register_component(Component(
        name="vllm:Qwen3VL",
        category="model",
        description="vLLM's built-in Qwen3-VL vision-language model",
        module_path="vllm.model_executor.models.qwen3_vl",
        class_name="Qwen3VLForConditionalGeneration",
        requires_custom_encoder=False,
        tags=["vision-language", "built-in"],
    ))
    register_component(Component(
        name="vllm:Qwen2Model",
        category="model",
        description="vLLM's built-in Qwen2 (compatible with Qwen3 decoder models)",
        module_path="vllm.model_executor.models.qwen2",
        class_name="Qwen2Model",
        requires_custom_encoder=False,
        tags=["decoder-only", "built-in"],
    ))
    register_component(Component(
        name="vllm:LFM2VL",
        category="model",
        description="vLLM's built-in LFM2-VL vision-language model (lightweight)",
        module_path="vllm.model_executor.models.lfm2_vl",
        class_name="Lfm2VLForConditionalGeneration",
        requires_custom_encoder=False,
        tags=["vision-language", "built-in", "lightweight"],
    ))

    # ── Poolers ──
    register_component(Component(
        name="colbert",
        category="pooler",
        description="ColBERT token-level pooler — multi-vector embeddings + L2 norm",
        module_path="poolers.colbert",
        class_name="ColBERTPooler",
        compatible_with=["modernbert", "vllm:ModernBertModel"],
        tags=["retrieval", "multi-vector", "MaxSim"],
    ))
    register_component(Component(
        name="gliner_span",
        category="pooler",
        description="GLiNER span pooler — LSTM + span markers for entity extraction",
        module_path="poolers.gliner",
        class_name="GLiNERSpanPooler",
        compatible_with=["modernbert", "vllm:ModernBertModel", "mt5"],
        tags=["NER", "span-extraction", "zero-shot"],
    ))
    register_component(Component(
        name="colpali",
        category="pooler",
        description="ColPali multi-vector pooler — for visual document retrieval",
        module_path="poolers.colpali",
        class_name="ColPaliPooler",
        compatible_with=["vllm:Qwen3VL", "vllm:LFM2VL"],
        tags=["retrieval", "multi-vector", "vision"],
    ))

    # ── Kernels ──
    for kernel_name, desc in [
        ("fused_glu_mlp", "Fused GLU MLP (GELU * gate in one Triton kernel)"),
        ("fused_layernorm", "Fused LayerNorm with optional bias"),
        ("fused_rope_global", "Fused global RoPE embedding"),
        ("fused_rope_local", "Fused local RoPE for sliding window attention"),
        ("fused_dropout_residual", "Fused dropout + residual addition"),
        ("flash_attention_rpb", "Flash Attention with T5-style relative position bias"),
        ("ff_fused", "Fused feedforward (GELU * MUL * dropout)"),
    ]:
        register_component(Component(
            name=kernel_name,
            category="kernel",
            description=desc,
            module_path=f"kernels.{kernel_name}",
            tags=["triton", "performance"],
        ))


# Auto-register on import
_register_builtins()
