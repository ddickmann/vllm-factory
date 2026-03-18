"""
GLiNER L4 reranker — Custom vLLM-optimized ModernBERT encoder + GLiNER projection,
LSTM, scorer.

Uses the same custom ModernBERT encoder as mmbert_gliner (SDPA attention, dual HF
RoPE, block-diagonal masks) instead of vLLM's built-in ModernBertModel — the built-in
uses vLLM's RoPE kernel which introduces numerical divergence on borderline entities.

Mirrors ``gliner.modeling.base.UniEncoderTokenModel`` inference stack:
  token_rep_layer (ModernBERT) → projection 512→768 → extract prompts/words (pooler)
  → LSTM on words → Scorer → (W, C, 3) logits.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from vllm.config import VllmConfig

from .config import GLiNERRerankConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import the custom ModernBERT encoder (same one used by mmbert_gliner).
# This encoder uses HF RoPE + SDPA + block-diagonal masks for numerical
# parity with the vanilla GLiNER library.
# ---------------------------------------------------------------------------
_ENCODER_PATH = Path(__file__).resolve().parents[2] / "models" / "modernbert" / "modernbert_encoder.py"


def _import_modernbert_encoder():
    spec = importlib.util.spec_from_file_location(
        "modernbert_encoder", str(_ENCODER_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("modernbert_encoder", mod)
    spec.loader.exec_module(mod)
    return mod


_modernbert_encoder_mod = _import_modernbert_encoder()
ModernBertModel = _modernbert_encoder_mod.ModernBertModel


def _patch_modernbert_config(cfg):
    """Ensure all attributes the custom ModernBertModel encoder expects are present."""
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

    if not hasattr(cfg, "norm_eps"):
        cfg.norm_eps = float(getattr(cfg, "layer_norm_eps", 1e-5))

    # GLiNERRerankConfig.__getattribute__ intercepts rope_theta and returns None
    # when nested rope_parameters (Transformers v5 format) is present. The custom
    # encoder reads global_rope_theta / local_rope_theta directly and sets
    # rope_theta on config copies, so we remove the nested dict to unblock access.
    try:
        rp = object.__getattribute__(cfg, "rope_parameters")
    except AttributeError:
        rp = None
    if isinstance(rp, dict) and any(
        k in rp for k in ("full_attention", "sliding_attention")
    ):
        try:
            delattr(cfg, "rope_parameters")
        except AttributeError:
            pass

    return cfg


class GLiNERRerankModel(nn.Module):
    """Pooling model: custom ModernBERT encoder → 768-d projection; pooler does RNN + scorer."""

    is_pooling_model = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg: GLiNERRerankConfig = vllm_config.model_config.hf_config
        cfg = _patch_modernbert_config(cfg)
        self.config = cfg
        self.vllm_config = vllm_config

        self.backbone = ModernBertModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.backbone" if prefix else "backbone",
        )

        enc_h = cfg.hidden_size
        gliner_h = cfg.gliner_hidden_size
        self.token_projection = nn.Linear(enc_h, gliner_h)
        self.rnn = nn.LSTM(
            input_size=gliner_h,
            hidden_size=gliner_h // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        H = gliner_h
        self.scorer_proj_token = nn.Linear(H, H * 2)
        self.scorer_proj_label = nn.Linear(H, H * 2)
        self.scorer_out_mlp = nn.Sequential(
            nn.Linear(H * 3, H * 4),
            nn.Dropout(0.0),
            nn.ReLU(),
            nn.Linear(H * 4, 3),
        )

        from .pooler import GLiNERRerankPooler

        self.pooler = GLiNERRerankPooler(self)

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.backbone.embeddings.tok_embeddings(input_ids)

    def sample(self, logits: torch.Tensor, sampling_metadata):
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
        with torch.no_grad():
            if input_ids is None:
                raise ValueError("GLiNERRerankModel requires input_ids")
            output = self.backbone(
                input_ids=input_ids,
                position_ids=positions.unsqueeze(0) if positions is not None and positions.dim() == 1 else positions,
                inputs_embeds=inputs_embeds,
            )
            hidden_states = (
                output.last_hidden_state
                if hasattr(output, "last_hidden_state")
                else output if isinstance(output, torch.Tensor) else output[0]
            )
            if hidden_states.dim() == 3:
                hidden_states = hidden_states.squeeze(0)
            h = self.token_projection(hidden_states)
        return h

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Map GLiNER HF checkpoint weights to backbone + head layers.

        HF prefix mapping:
            token_rep_layer.bert_layer.model.*  ->  backbone (self.backbone)
            token_rep_layer.projection.*        ->  self.token_projection
            rnn.lstm.*                          ->  self.rnn
            scorer.*                            ->  self.scorer_*

        Uses load_state_dict for backbone (bypasses VocabParallelEmbedding
        weight_loader assertion that can fail when vLLM adjusts vocab_size).
        """
        backbone_prefix = "token_rep_layer.bert_layer.model."
        proj_prefix = "token_rep_layer.projection."

        backbone_state = {}
        proj_state = {}
        rnn_state = {}
        scorer_state = {}

        vllm_backbone = self.backbone.state_dict()

        for hf_name, tensor in weights:
            if hf_name.startswith(backbone_prefix):
                hf_key = hf_name[len(backbone_prefix):]

                if hf_key not in vllm_backbone:
                    continue

                if "tok_embeddings.weight" in hf_key:
                    target_shape = vllm_backbone[hf_key].shape
                    if tensor.shape[0] < target_shape[0]:
                        pad = torch.zeros(
                            target_shape[0] - tensor.shape[0], tensor.shape[1],
                            dtype=tensor.dtype, device=tensor.device,
                        )
                        tensor = torch.cat([tensor, pad], dim=0)
                    elif tensor.shape[0] > target_shape[0]:
                        tensor = tensor[:target_shape[0], :]

                backbone_state[hf_key] = tensor
            elif hf_name.startswith(proj_prefix):
                proj_state[hf_name[len(proj_prefix):]] = tensor
            elif hf_name.startswith("rnn.lstm."):
                rnn_state[hf_name[len("rnn.lstm."):]] = tensor
            elif hf_name.startswith("scorer."):
                scorer_state[hf_name[len("scorer."):]] = tensor

        self.backbone.load_state_dict(backbone_state, strict=False)
        logger.info("ModernBERT backbone: loaded %d/%d keys", len(backbone_state), len(vllm_backbone))

        if proj_state:
            self.token_projection.load_state_dict(
                {"weight": proj_state["weight"], "bias": proj_state["bias"]},
                strict=True,
            )
            logger.info("token_projection loaded")

        if rnn_state:
            self.rnn.load_state_dict(rnn_state, strict=False)
            logger.info("LSTM loaded")

        if scorer_state:
            for key, tensor in scorer_state.items():
                if key.startswith("proj_token."):
                    getattr(self.scorer_proj_token, key[len("proj_token."):]).data.copy_(tensor)
                elif key.startswith("proj_label."):
                    getattr(self.scorer_proj_label, key[len("proj_label."):]).data.copy_(tensor)
                elif key.startswith("out_mlp."):
                    parts = key.split(".")
                    idx = int(parts[1])
                    attr_name = parts[2]
                    getattr(self.scorer_out_mlp[idx], attr_name).data.copy_(tensor)
            logger.info("Scorer loaded")

        return set(n for n, _ in self.named_parameters())
