"""
MT5-based GLiNER model for vLLM — zero-shot NER via span extraction.

Architecture:
    Custom MT5Encoder (flash_attention_rpb + fused_gelu_mul_dropout Triton kernels
    + vLLM parallel layers) → projection (d_model→hidden_size) → GLiNER span pooler

Backbone: models.mt5.MT5Encoder — HF-oriented T5 encoder with Triton kernel replacements
          for attention (flash_attention_rpb) and feed-forward (fused_gelu_mul_dropout).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from poolers.gliner import GLiNERSpanPooler

from .config import GLiNERMT5Config

# Load the custom MT5 encoder with Triton kernels
_ENCODER_PATH = Path(__file__).resolve().parents[2] / "models" / "mt5" / "mt5_encoder.py"


def _import_mt5_encoder():
    spec = importlib.util.spec_from_file_location(
        "mt5_encoder", str(_ENCODER_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("mt5_encoder", mod)
    spec.loader.exec_module(mod)
    return mod


_encoder_mod = _import_mt5_encoder()
MT5Encoder = _encoder_mod.MT5Encoder


# HF T5 key → custom MT5Encoder key mapping
_ATTN_PROJ_MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "out_proj"}

_BLOCK_ATTN_RE = re.compile(
    r"^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.(.+)$"
)
_BLOCK_ATTN_NORM_RE = re.compile(
    r"^encoder\.block\.(\d+)\.layer\.0\.layer_norm\.(.+)$"
)
_BLOCK_FF_RE = re.compile(
    r"^encoder\.block\.(\d+)\.layer\.1\.DenseReluDense\.(.+)$"
)
_BLOCK_FF_NORM_RE = re.compile(
    r"^encoder\.block\.(\d+)\.layer\.1\.layer_norm\.(.+)$"
)


def _map_hf_t5_key(hf_key: str) -> str | None:
    """Map an HF T5EncoderModel state_dict key to the custom MT5Encoder param name.

    Returns None if the key cannot be mapped (e.g. decoder-only keys).
    """
    # embed_tokens (HF stores as either shared.weight or encoder.embed_tokens.weight)
    if hf_key in ("shared.weight", "encoder.embed_tokens.weight"):
        return "embed_tokens.weight"

    # final_layer_norm
    if hf_key.startswith("encoder.final_layer_norm."):
        suffix = hf_key[len("encoder.final_layer_norm."):]
        return f"final_ln.{suffix}"

    # Attention projections + RPB
    m = _BLOCK_ATTN_RE.match(hf_key)
    if m:
        layer_idx, remainder = m.group(1), m.group(2)
        parts = remainder.split(".", 1)
        proj_name = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        if proj_name == "relative_attention_bias":
            return f"layers.{layer_idx}.self_attn.rpb.relative_attention_bias.{suffix}" if suffix else None
        elif proj_name in _ATTN_PROJ_MAP:
            mapped_proj = _ATTN_PROJ_MAP[proj_name]
            return f"layers.{layer_idx}.self_attn.{mapped_proj}.{suffix}" if suffix else f"layers.{layer_idx}.self_attn.{mapped_proj}"
        return None

    # Attention layer norm → ln1
    m = _BLOCK_ATTN_NORM_RE.match(hf_key)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        return f"layers.{layer_idx}.ln1.{suffix}"

    # Feed-forward
    m = _BLOCK_FF_RE.match(hf_key)
    if m:
        layer_idx, remainder = m.group(1), m.group(2)
        return f"layers.{layer_idx}.ff.{remainder}"

    # FF layer norm → ln2
    m = _BLOCK_FF_NORM_RE.match(hf_key)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        return f"layers.{layer_idx}.ln2.{suffix}"

    return None


class GLiNERMT5Model(nn.Module):
    """MT5-based GLiNER for span classification.

    Pipeline:
        input_ids → MT5Encoder(d_model) → projection(d_model→hidden_size)
                  → GLiNERSpanPooler (LSTM + SpanMarker + einsum)
                  → span scores
    """

    is_pooling_model = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg: GLiNERMT5Config = vllm_config.model_config.hf_config
        self.config = cfg
        self.vllm_config = vllm_config

        self.d_model = int(getattr(cfg, "d_model", 1024))
        self.pooler_hidden_size = int(getattr(cfg, "gliner_hidden_size", 768))

        # 1. Backbone — custom MT5Encoder with Triton kernels
        cache_config = getattr(vllm_config, "cache_config", None)
        quant_config = getattr(vllm_config, "quant_config", None)
        self.backbone = MT5Encoder(
            cfg=cfg,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix="encoder",
        )

        # 2. Projection layer (d_model → hidden_size) if dimensions differ
        if self.d_model != self.pooler_hidden_size:
            self.projection = nn.Linear(self.d_model, self.pooler_hidden_size)
        else:
            self.projection = None

        # 3. GLiNER span pooler (shared implementation)
        self.pooler = GLiNERSpanPooler(cfg)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM's pooling runner for embedding lookup."""
        return self.backbone.embed_tokens(input_ids)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        """Override sampling for pooling models — return empty outputs."""
        try:
            from vllm.sequence import SamplerOutput
            return SamplerOutput(outputs=[])
        except ImportError:
            return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run MT5 encoder + projection, return hidden states for pooler."""
        if input_ids is not None and input_ids.dim() == 1:
            input_ids_2d = input_ids.unsqueeze(0)
        else:
            input_ids_2d = input_ids

        # Build attention mask (all ones — no padding within vLLM sequences)
        attention_mask = torch.ones(
            input_ids_2d.shape, dtype=torch.long, device=input_ids_2d.device
        )

        with torch.no_grad():
            hs = self.backbone(
                input_ids=input_ids_2d,
                attention_mask=attention_mask,
            )

        # Custom encoder returns tensor directly
        if hs.dim() == 3:
            hs = hs.squeeze(0)

        if self.projection is not None:
            hs = self.projection(hs)

        return hs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load backbone via weight mapping, then GLiNER head weights from checkpoint.

        Maps HF T5 state_dict keys to custom MT5Encoder param names and loads
        via vLLM's default_weight_loader for tensor-parallel correctness.
        """
        gliner_prefix = "token_rep_layer.bert_layer.model."
        projection_prefix = "token_rep_layer.projection."
        rnn_prefix = "rnn."
        span_rep_prefix = "span_rep_layer.span_rep_layer."
        prompt_rep_prefix = "prompt_rep_layer."

        backbone_weights = []
        projection_state = {}
        pooler_weights = {}

        vllm_pooler = self.pooler.state_dict()

        for hf_name, tensor in weights:
            if hf_name.startswith(gliner_prefix):
                hf_key = hf_name[len(gliner_prefix):]
                mapped = _map_hf_t5_key(hf_key)
                if mapped is not None:
                    backbone_weights.append((mapped, tensor))

            elif hf_name.startswith(projection_prefix):
                stripped = hf_name[len(projection_prefix):]
                projection_state[stripped] = tensor

            else:
                vllm_key = hf_name
                if hf_name.startswith(rnn_prefix):
                    vllm_key = hf_name
                elif hf_name.startswith(span_rep_prefix):
                    vllm_key = hf_name.replace(span_rep_prefix, "span_rep.")
                elif hf_name.startswith(prompt_rep_prefix):
                    vllm_key = hf_name.replace(prompt_rep_prefix, "prompt_proj.")

                if vllm_key in vllm_pooler:
                    pooler_weights[vllm_key] = tensor

        # Load backbone via vLLM weight loading (handles ColumnParallelLinear etc.)
        params_dict = dict(self.backbone.named_parameters())
        loaded_count = 0
        for name, loaded_weight in backbone_weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_count += 1

        print(f"[GLiNERMT5] Loaded backbone: {loaded_count}/{len(params_dict)} params")

        # Load projection
        device = next(self.backbone.parameters()).device
        dtype = self.vllm_config.model_config.dtype
        if self.projection is not None and projection_state:
            self.projection.load_state_dict(projection_state)
            self.projection.to(device=device, dtype=dtype)
            print(f"[GLiNERMT5] Loaded projection: {list(projection_state.keys())}")

        # Load pooler
        if pooler_weights:
            self.pooler.load_state_dict(pooler_weights, strict=False)
            self.pooler.to(device=device, dtype=dtype)
            print(f"[GLiNERMT5] Loaded pooler: {len(pooler_weights)}/{len(vllm_pooler)} keys")
        else:
            print("[GLiNERMT5] WARNING: No pooler weights loaded!")
            print(f"  Available prefixes: {set(k.split('.')[0] for k in pooler_weights.keys())}")

        return set(name for name, _ in self.named_parameters())
