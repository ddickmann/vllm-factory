"""
Triton Kernels for vLLM Factory.

High-performance fused operations for custom model plugins.
Import from here — each kernel provides graceful fallbacks when Triton
is unavailable.

Available kernels:
    - fused_glu_mlp: Fused GELU * gate MLP
    - fused_layernorm: Fused LayerNorm with optional bias
    - fused_rope_global: Fused global RoPE embedding
    - fused_rope_local: Fused local (sliding window) RoPE
    - fused_dropout_residual: Fused dropout + residual add
    - flash_attention_rpb: Flash Attention with T5 relative position bias
    - ff_fused: Fused feedforward (GELU * MUL * dropout)
"""
