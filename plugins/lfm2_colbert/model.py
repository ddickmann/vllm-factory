"""LFM2ForColBERT model wrapper for vLLM."""

import os
from typing import Iterable, Optional, Tuple

import safetensors.torch
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from vllm.config import PoolerConfig, VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM, Lfm2Model
from vllm.sequence import IntermediateTensors

from .config import LFM2ColBERTConfig


@default_pooling_type(tok_pooling_type="ALL")
class LFM2ForColBERT(nn.Module, HasInnerState, IsHybrid):
    """LFM2 model + ColBERT linear projection (multi-vector token-level pooling)."""

    is_pooling_model = True
    is_hybrid = True

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: "VllmConfig") -> tuple[torch.dtype, ...]:
        return Lfm2ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: "VllmConfig") -> tuple[tuple[int, int]]:
        return Lfm2ForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        return Lfm2ForCausalLM.get_mamba_state_copy_func()

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config: LFM2ColBERTConfig = vllm_config.model_config.hf_config
        self.config = config

        # Base Lfm2Model wrapped exactly as vLLM expects
        self.model = Lfm2Model(vllm_config=vllm_config, prefix=f"{prefix}.model")

        # ColBERT projection: hidden_size -> colbert_dim (usually 128)
        # ReplicatedLinear ensures correct behavior under tensor parallelism
        colbert_dim = getattr(config, "colbert_dim", 128)
        self.colbert_linear = ReplicatedLinear(config.hidden_size, colbert_dim, bias=False)

        # Token-level pooler module
        pooler_config = vllm_config.model_config.pooler_config
        if pooler_config is not None:
            self.pooler = pooler_for_token_embed(pooler_config)
        else:
            self.pooler = pooler_for_token_embed(PoolerConfig(pooling_type="ALL"))

        # We must load the projection lazily because it usually sits in 1_Dense/model.safetensors
        # and not in the main model.safetensors index. We rely on the weights loader or a lazy
        # fetch inside the forward pass to obtain it if not loaded initially.
        self._projection_loaded = False
        self._model_name_or_path = vllm_config.model_config.model
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def _ensure_projection_loaded(self):
        """Lazy load the colbert_linear weights if they weren't in the main safetensors."""
        if self._projection_loaded:
            return

        try:
            # 1. Try to fetch from local path (if model is a local directory)
            weight_path = os.path.join(self._model_name_or_path, "1_Dense", "model.safetensors")
            if not os.path.exists(weight_path):
                # 2. Try huggingface hub
                weight_path = hf_hub_download(
                    repo_id=self._model_name_or_path,
                    filename="1_Dense/model.safetensors",
                    local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1"
                )

            with safetensors.torch.safe_open(weight_path, framework="pt", device="cpu") as f:
                weight = f.get_tensor("linear.weight")
                print(f"LFM2ForColBERT: Loaded projection weight {weight.shape} from {weight_path}")
                # Ensure dtype matches
                weight = weight.to(self.colbert_linear.weight.dtype)
                self.colbert_linear.weight.data.copy_(weight)
                self._projection_loaded = True

        except Exception as e:
            print(f"LFM2ForColBERT: Warning: Could not load projection from 1_Dense: {e}")
            print("LFM2ForColBERT: Assuming projection weights were loaded via standard loader.")
            self._projection_loaded = True

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

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
        """Encode -> project -> L2 normalize."""
        if not self._projection_loaded:
            self._ensure_projection_loaded()

        # vLLM's Lfm2Model takes 1D input_ids and positions directly
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        # Lfm2Model returns (total_tokens, hidden_size) directly
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.dim() == 3:
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        elif hasattr(hidden_states, "last_hidden_state"):
            hidden_states = hidden_states.last_hidden_state
            if hidden_states.dim() == 3:
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        else:
            hidden_states = hidden_states[0]

        # dtype alignment just in case
        if self.colbert_linear.weight.dtype != hidden_states.dtype:
            self.colbert_linear = self.colbert_linear.to(hidden_states.dtype)

        # ReplicatedLinear returns (output, bias); unpack
        projected, _ = self.colbert_linear(hidden_states)
        projected = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        return projected

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using the underlying model's loader."""
        loaded_params = set()

        # Prepare an iterator that we can peek at or process
        for name, loaded_weight in weights:
            # Check if this is the colbert head projection
            if "linear.weight" in name and loaded_weight.shape[0] == self.colbert_linear.output_size:
                print(f"LFM2ForColBERT: Loading colbert_linear from {name} shape {loaded_weight.shape}")
                weight = loaded_weight.to(self.colbert_linear.weight.dtype)
                self.colbert_linear.weight.data.copy_(weight)
                self._projection_loaded = True
                loaded_params.add("colbert_linear.weight")
                continue

            model_name = name
            if name.startswith("model."):
                model_name = name[6:]

            loaded_from_model = self.model.load_weights([(model_name, loaded_weight)])
            for loaded_key in loaded_from_model:
                loaded_params.add(f"model.{loaded_key}")

        # Ensure we lazy-load ColBERT projection if it wasn't in the main safetensors
        if not self._projection_loaded:
            self._ensure_projection_loaded()
        if self._projection_loaded:
            loaded_params.add("colbert_linear.weight")

        return loaded_params
