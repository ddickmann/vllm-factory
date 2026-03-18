"""
Weight loading utilities for vLLM model plugins.

Common patterns for mapping checkpoint keys to model parameters,
loading weights from separate files (e.g., projection layers), and
handling safetensors files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn


def map_weight_name(
    name: str,
    prefix_mapping: dict[str, str] | None = None,
    strip_prefixes: list[str] | None = None,
    add_prefix: str | None = None,
) -> str:
    """Map a checkpoint weight name to the model's parameter name.

    Args:
        name: Original weight name from checkpoint
        prefix_mapping: Dict of {old_prefix: new_prefix} for substitution
        strip_prefixes: List of prefixes to strip
        add_prefix: Prefix to add if not already present

    Returns:
        Mapped parameter name
    """
    # Apply explicit prefix mapping first
    if prefix_mapping:
        for old, new in prefix_mapping.items():
            if name.startswith(old):
                name = new + name[len(old):]
                return name

    # Strip prefixes
    if strip_prefixes:
        for prefix in strip_prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

    # Add prefix
    if add_prefix and not name.startswith(add_prefix):
        name = add_prefix + name

    return name


def load_weights_from_mapping(
    model: nn.Module,
    weights: Iterable[Tuple[str, torch.Tensor]],
    prefix_mapping: dict[str, str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> set[str]:
    """Load weights into a model with name mapping.

    Args:
        model: The target nn.Module
        weights: Iterator of (name, tensor) pairs
        prefix_mapping: Dict of {checkpoint_prefix: model_prefix}
        ignore_patterns: List of substrings — skip weights containing these

    Returns:
        Set of parameter names that were successfully loaded
    """
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    params_dict = dict(model.named_parameters())
    loaded = set()

    for name, loaded_weight in weights:
        # Skip ignored patterns
        if ignore_patterns and any(p in name for p in ignore_patterns):
            continue

        # Map the name
        mapped = map_weight_name(name, prefix_mapping=prefix_mapping)

        # Try to find the parameter
        if mapped in params_dict:
            param = params_dict[mapped]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(mapped)

    return loaded


def load_safetensors_file(
    model_path: str,
    relative_path: str,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load tensors from a safetensors file, with HuggingFace Hub fallback.

    If the file doesn't exist locally and model_path looks like a
    HuggingFace Hub ID, downloads the file automatically.

    Args:
        model_path: Local directory or HuggingFace model ID
        relative_path: Path relative to model_path (e.g., "1_Dense/model.safetensors")
        device: Device to load tensors to

    Returns:
        Dict of {key: tensor}

    Raises:
        FileNotFoundError: If file can't be found or downloaded
    """
    from safetensors import safe_open

    local_file = Path(model_path) / relative_path

    # Try HuggingFace Hub if not local
    if not local_file.exists() and "/" in str(model_path) and not Path(model_path).is_dir():
        try:
            from huggingface_hub import hf_hub_download

            hf_token = os.environ.get("HF_TOKEN")
            downloaded = hf_hub_download(
                repo_id=model_path,
                filename=relative_path,
                token=hf_token,
            )
            local_file = Path(downloaded)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find or download '{relative_path}' from '{model_path}': {e}"
            ) from e

    if not local_file.exists():
        raise FileNotFoundError(f"File not found: {local_file}")

    tensors = {}
    with safe_open(str(local_file), framework="pt", device=device) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    return tensors
