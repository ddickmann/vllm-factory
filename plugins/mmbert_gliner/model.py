"""
mmBERT-GLiNER — Custom vLLM-optimized ModernBERT encoder + GLiNER span extraction pooler.

Backbone: models.modernbert.ModernBertModel (4 fused Triton kernels + vLLM parallel layers)
Pooler:   poolers.gliner (LSTM + SpanMarker + einsum scoring)
Weights:  GLiNER HF checkpoint with token_rep_layer.bert_layer.model.* prefix

vLLM 0.15.x compatible:
- num_hidden_layers=0 in config to skip KV cache allocation
- Custom encoder uses its own attention (SDPA + fused RoPE/GLU/LayerNorm/Dropout Triton kernels)
- Forward returns 2D (total_tokens, hidden_size) for the pooler
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import nn
from vllm.config import VllmConfig

from poolers.gliner import GLiNERSpanPooler
from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

logger = logging.getLogger(__name__)

_ENCODER_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "modernbert" / "modernbert_encoder.py"
)


def _import_modernbert_encoder():
    spec = importlib.util.spec_from_file_location("modernbert_encoder", str(_ENCODER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("modernbert_encoder", mod)
    spec.loader.exec_module(mod)
    return mod


_modernbert_encoder_mod = _import_modernbert_encoder()
ModernBertModel = _modernbert_encoder_mod.ModernBertModel


def _patch_modernbert_config(cfg):
    """Ensure all attributes the custom ModernBertModel encoder expects are present.

    GLiNERModernBertConfig extends PretrainedConfig (NOT ModernBertConfig),
    so it may be missing dropout, bias, and norm attributes that the custom
    encoder's layer constructors access directly.  We set safe defaults for
    anything absent.
    """
    # The config already carries the real encoder geometry (encoder_num_layers,
    # hidden_size, global_attn_every_n_layers, etc.) — we only need to add
    # attributes that GLiNERModernBertConfig doesn't define but the custom
    # encoder's layer constructors read directly (dropout variants, bias flags).
    for attr, default in (
        ("embedding_dropout", 0.0),
        ("mlp_dropout", 0.0),
        ("attention_dropout", 0.0),
        ("classifier_dropout", 0.0),
        ("norm_bias", False),
        ("attention_bias", False),
        ("mlp_bias", False),
        ("classifier_bias", False),
    ):
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)

    # norm_eps: might be stored under a different name
    if not hasattr(cfg, "norm_eps"):
        cfg.norm_eps = float(getattr(cfg, "layer_norm_eps", 1e-5))

    return cfg


class GLiNERModernBertModel(nn.Module):
    """Custom vLLM-optimized ModernBERT encoder + GLiNER span extraction pooler.

    Architecture:
        input_ids -> ModernBertModel (fused Triton kernels + vLLM parallel layers)
                  -> (total_tokens, hidden_size)
                  -> GLiNERSpanPooler (LSTM + SpanMarker + einsum)
                  -> span scores
    """

    is_pooling_model = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config

        cfg = _patch_modernbert_config(cfg)

        self.config = cfg
        self.vllm_config = vllm_config

        # Temporarily set num_hidden_layers and num_attention_heads to the real
        # encoder values so the custom encoder constructs correctly. Restore
        # them afterwards so vLLM's scheduler sees 0 layers / no KV cache.
        saved_nhl = cfg.num_hidden_layers
        saved_nah = cfg.num_attention_heads
        cfg.num_hidden_layers = cfg.encoder_num_layers
        cfg.num_attention_heads = cfg.encoder_num_attention_heads

        self.model = ModernBertModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        cfg.num_hidden_layers = saved_nhl
        cfg.num_attention_heads = saved_nah

        encoder_h = int(cfg.hidden_size)
        gliner_h = int(getattr(cfg, "gliner_hidden_size", encoder_h))
        if encoder_h != gliner_h:
            self.projection = nn.Linear(encoder_h, gliner_h)
        else:
            self.projection = None

        self._business_pooler = GLiNERSpanPooler(cfg)
        self.pooler = VllmPoolerAdapter(self._business_pooler, requires_token_ids=True)
        self.pooler.to(vllm_config.model_config.dtype)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM's pooling runner for embedding lookup."""
        return self.model.embeddings.tok_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns encoder hidden states; pooler is called by vLLM pooling runner."""
        output = self.model(
            input_ids=input_ids,
            position_ids=positions.unsqueeze(0)
            if positions is not None and positions.dim() == 1
            else positions,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output
            if isinstance(output, torch.Tensor)
            else output[0]
        )

        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)

        if self.projection is not None:
            hidden_states = self.projection(hidden_states)

        return hidden_states

    def sample(self, logits: torch.Tensor, sampling_metadata):
        """Override sampling for pooling models — return empty outputs."""
        try:
            from vllm.sequence import SamplerOutput

            return SamplerOutput(outputs=[])
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Map GLiNER HF checkpoint weights to backbone + projection + pooler.

        HF prefix mapping:
            token_rep_layer.bert_layer.model.*  ->  backbone (self.model)
            token_rep_layer.projection.*        ->  self.projection (encoder→gliner dim)
            rnn.*                               ->  pooler.rnn.*
            span_rep_layer.span_rep_layer.*     ->  pooler.span_rep.*
            prompt_rep_layer.*                  ->  pooler.prompt_proj.*
        """
        backbone_prefix = "token_rep_layer.bert_layer.model."
        projection_prefix = "token_rep_layer.projection."
        rnn_prefix = "rnn."
        span_rep_prefix = "span_rep_layer.span_rep_layer."
        prompt_rep_prefix = "prompt_rep_layer."

        backbone_state = {}
        pooler_state = {}
        projection_state = {}

        vllm_backbone = self.model.state_dict()
        pooler_keys = set(self._business_pooler.state_dict().keys())
        vllm_projection = self.projection.state_dict() if self.projection is not None else {}

        for hf_name, tensor in weights:
            if hf_name.startswith(backbone_prefix):
                hf_key = hf_name[len(backbone_prefix) :]

                if hf_key not in vllm_backbone:
                    continue

                if "tok_embeddings.weight" in hf_key:
                    target_shape = vllm_backbone[hf_key].shape
                    if tensor.shape[0] < target_shape[0]:
                        pad = torch.zeros(
                            target_shape[0] - tensor.shape[0],
                            tensor.shape[1],
                            dtype=tensor.dtype,
                            device=tensor.device,
                        )
                        tensor = torch.cat([tensor, pad], dim=0)
                    elif tensor.shape[0] > target_shape[0]:
                        tensor = tensor[: target_shape[0], :]

                backbone_state[hf_key] = tensor
            elif hf_name.startswith(projection_prefix):
                local_key = hf_name[len(projection_prefix) :]
                if local_key in vllm_projection:
                    projection_state[local_key] = tensor
            else:
                vllm_key = hf_name
                if hf_name.startswith(rnn_prefix):
                    vllm_key = hf_name
                elif hf_name.startswith(span_rep_prefix):
                    vllm_key = hf_name.replace(span_rep_prefix, "span_rep.")
                elif hf_name.startswith(prompt_rep_prefix):
                    vllm_key = hf_name.replace(prompt_rep_prefix, "prompt_proj.")

                if vllm_key in pooler_keys:
                    pooler_state[vllm_key] = tensor

        self.model.load_state_dict(backbone_state, strict=False)
        logger.info(
            "[mmBERT-GLiNER] Loaded backbone: %s/%s keys",
            len(backbone_state),
            len(vllm_backbone),
        )

        if self.projection is not None and projection_state:
            self.projection.load_state_dict(projection_state, strict=False)
            logger.info(
                "[mmBERT-GLiNER] Loaded projection: %s/%s keys",
                len(projection_state),
                len(vllm_projection),
            )

        self._business_pooler.load_state_dict(pooler_state, strict=False)
        logger.info(
            "[mmBERT-GLiNER] Loaded pooler: %s/%s keys",
            len(pooler_state),
            len(pooler_keys),
        )

        return set(name for name, _ in self.named_parameters())
