"""
ModernColBERT — Custom ModernBERT encoder + ColBERT multi-vector pooler.

Backbone: models.modernbert (custom encoder with fused Triton kernels)
Pooler:   ColBERT projection (nn.Linear → L2 norm), token-level
Weights:  Main encoder from HF checkpoint + projection from 1_Dense/model.safetensors

vLLM 0.15.x compatible:
- @attn_type("encoder_only") for bidirectional attention (no KV cache)
- @default_pooling_type(tok_pooling_type="ALL") for token-level embeddings
- embed_input_ids() for the pooling runner
"""

import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

# Load the shared ModernBERT encoder by absolute file path.
# This avoids sys.path manipulation and works in all contexts:
# editable .pth install, vLLM worker subprocess, direct script execution.
# Mirrors the superpod colbert_plugin self-contained encoder approach.
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

from vllm.config import VllmConfig  # noqa: E402
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed  # noqa: E402
from vllm.model_executor.model_loader.weight_utils import default_weight_loader  # noqa: E402
from vllm.model_executor.models.interfaces_base import attn_type, default_pooling_type  # noqa: E402
from vllm.sequence import IntermediateTensors  # noqa: E402

from .config import ModernColBERTConfig  # noqa: E402


@attn_type("encoder_only")
@default_pooling_type(tok_pooling_type="ALL")
class ModernBertForColBERT(nn.Module):
    """Custom ModernBERT encoder + ColBERT projection + L2 normalization.

    Architecture:
        input_ids → ModernBertModel → (total_tokens, hidden_size)
                  → colbert_linear  → (total_tokens, colbert_dim)
                  → L2 normalize    → (total_tokens, colbert_dim)
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        pooler_config = vllm_config.model_config.pooler_config

        # Ensure ColBERT parameters exist on config
        if not isinstance(config, ModernColBERTConfig):
            config.colbert_dim = getattr(config, "colbert_dim", 128)
            config.query_length = getattr(config, "query_length", 256)
            config.document_length = getattr(config, "document_length", 8192)

        self.config = config
        self.model_path = vllm_config.model_config.model
        self.colbert_dim = getattr(config, "colbert_dim", 128)
        self.hidden_size = config.hidden_size

        # 1. Backbone — custom ModernBERT with Triton kernels
        self.model = ModernBertModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        # 2. ColBERT projection — Linear(hidden → colbert_dim), no bias
        self.colbert_linear = nn.Linear(
            config.hidden_size,
            self.colbert_dim,
            bias=False,
        )

        # 3. vLLM pooler for token-level embedding task
        if pooler_config is not None:
            self.pooler = pooler_for_token_embed(pooler_config)
        else:
            from vllm.config import PoolerConfig

            self.pooler = pooler_for_token_embed(PoolerConfig(pooling_type="ALL"))

        self._projection_loaded = False

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM pooling runner for embeddings lookup."""
        return self.model.embeddings.tok_embeddings(input_ids)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        try:
            from vllm.sequence import SamplerOutput
            return SamplerOutput(outputs=[])
        except ImportError:
            return None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode → project → L2 normalize.

        Returns: (total_tokens, colbert_dim) tensor of unit-length embeddings.
        """
        if not self._projection_loaded:
            self._ensure_projection_loaded()

        # vLLM sends 1D tensors; encoder expects 2D
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if positions is not None and positions.dim() == 1:
            positions = positions.unsqueeze(0)

        output = self.model(
            input_ids=input_ids,
            position_ids=positions,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output if isinstance(output, torch.Tensor) else output[0]
        )

        # Flatten to (total_tokens, hidden_size) for token-level pooling
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Dtype alignment
        if self.colbert_linear.weight.dtype != hidden_states.dtype:
            self.colbert_linear = self.colbert_linear.to(hidden_states.dtype)

        # Project + L2 normalize (critical for MaxSim)
        projected = self.colbert_linear(hidden_states)
        projected = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return projected

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        """Load encoder weights + ColBERT projection from 1_Dense/model.safetensors."""
        weights_list = list(weights)
        loaded = set()
        params = dict(self.named_parameters())

        model_weights = []
        projection_weights = {}

        for name, param in weights_list:
            if any(k in name for k in ("1_Dense", "colbert_linear", "projection", "dense")):
                projection_weights[name] = param
            else:
                model_weights.append((name, param))

        # Encoder weights
        for name, loaded_weight in model_weights:
            mapped = name
            if not name.startswith("model."):
                mapped = "model." + name
            if name.startswith("encoder."):
                mapped = "model." + name[8:]

            if mapped in params:
                weight_loader = getattr(params[mapped], "weight_loader", default_weight_loader)
                weight_loader(params[mapped], loaded_weight)
                loaded.add(mapped)

        # Projection weights
        for name, param in projection_weights.items():
            if ("weight" in name.lower() or param.dim() == 2) and "colbert_linear.weight" in params:
                target = params["colbert_linear.weight"]
                if target.shape == param.shape:
                    weight_loader = getattr(target, "weight_loader", default_weight_loader)
                    weight_loader(target, param)
                    loaded.add("colbert_linear.weight")

        # Fallback: load projection from separate file (1_Dense/model.safetensors)
        if "colbert_linear.weight" not in loaded:
            if self._load_projection_from_file():
                loaded.add("colbert_linear.weight")

        self._projection_loaded = True
        return loaded

    def _ensure_projection_loaded(self):
        """Safety net: load projection on first forward if missing."""
        w = self.colbert_linear.weight.data
        if w.float().std().item() < 0.001 or w.float().std().item() > 0.5:
            self._load_projection_from_file()
        self._projection_loaded = True

    def _load_projection_from_file(self) -> bool:
        """Load ColBERT projection from 1_Dense/model.safetensors (local or HF Hub cache).

        Returns True if the weight was loaded successfully, False otherwise.
        """
        import os

        from safetensors import safe_open

        dense_file = Path(self.model_path) / "1_Dense" / "model.safetensors"

        if not dense_file.exists():
            # Try HF Hub (works with local cache even when HF_HUB_OFFLINE=1)
            try:
                from huggingface_hub import hf_hub_download
                dense_file = Path(
                    hf_hub_download(
                        repo_id=str(self.model_path),
                        filename="1_Dense/model.safetensors",
                        token=os.environ.get("HF_TOKEN"),
                    )
                )
            except Exception as e:
                print(f"[ModernColBERT] Could not locate 1_Dense/model.safetensors: {e}")
                return False

        if not dense_file.exists():
            print(f"[ModernColBERT] 1_Dense/model.safetensors not found at {dense_file}")
            return False

        print(f"[ModernColBERT] Loading ColBERT projection from {dense_file}")
        loaded_ok = False
        with safe_open(str(dense_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if ("weight" in key.lower() or tensor.dim() == 2) and \
                        self.colbert_linear.weight.shape == tensor.shape:
                    self.colbert_linear.weight.data.copy_(tensor)
                    print(f"[ModernColBERT] ✓ colbert_linear.weight loaded: {tensor.shape}")
                    loaded_ok = True
        return loaded_ok
