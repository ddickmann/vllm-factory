"""
Plumb PoolingParams.extra_kwargs[\"attention_mask\"] into model forward for pooling models.

vLLM's GpuModelRunner only merges ``compressed_token_type_ids`` from ``extra_kwargs`` into
``model_kwargs`` by default. For GLiNER-Linker, the collator's attention mask must reach
``DebertaEncoderModel`` so padded batches (tokenizer padding + vLLM max-prompt padding with
``vocab_size`` fillers) do not attend through pad positions.

This module monkey-patches ``GpuModelRunner._preprocess`` once (idempotent).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_gliner_linker_preprocess_patched"


def _build_flat_attention_mask(
    runner,
    scheduler_output: Any,
    num_input_tokens: int,
) -> Optional[torch.Tensor]:
    """Return (num_input_tokens,) int mask on runner.device, or None to use model defaults."""
    num_reqs = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids
    pooling_params = runner.input_batch.get_pooling_params()

    counts: list[int] = []
    for rid in req_ids:
        n = scheduler_output.num_scheduled_tokens.get(rid)
        if n is None:
            logger.warning("Missing num_scheduled_tokens for req %s; skip AM patch", rid)
            return None
        counts.append(int(n))

    pieces: list[torch.Tensor] = []
    for i in range(num_reqs):
        ek = pooling_params[i].extra_kwargs or {}
        full_m = ek.get("attention_mask")
        if full_m is None:
            return None
        npt = int(runner.input_batch.num_prompt_tokens[i])
        computed = int(runner.input_batch.num_computed_tokens_cpu[i])
        L = counts[i]
        if len(full_m) != npt:
            raise ValueError(
                f"attention_mask length {len(full_m)} must equal num_prompt_tokens={npt} "
                f"for request index {i}"
            )
        if computed + L > npt:
            raise ValueError(
                f"attention_mask slice out of range: computed={computed}, L={L}, npt={npt}"
            )
        sl = full_m[computed : computed + L]
        pieces.append(torch.tensor(sl, dtype=torch.long, device=runner.device))

    flat = torch.cat(pieces, dim=0)
    if flat.shape[0] > num_input_tokens:
        flat = flat[:num_input_tokens]
    elif flat.shape[0] < num_input_tokens:
        pad = torch.zeros(
            num_input_tokens - flat.shape[0],
            dtype=flat.dtype,
            device=flat.device,
        )
        flat = torch.cat([flat, pad], dim=0)
    return flat


def _make_patched_preprocess(orig_preprocess):
    def _preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors=None):
        out = orig_preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors)
        input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec = out
        if not getattr(self, "is_pooling_model", False) or input_ids is None:
            return out
        try:
            am = _build_flat_attention_mask(self, scheduler_output, int(input_ids.shape[0]))
        except Exception:
            logger.exception("GLiNER-Linker: failed to build attention_mask for pooling batch")
            raise
        if am is not None:
            model_kwargs = dict(model_kwargs)
            model_kwargs["attention_mask"] = am
        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec,
        )

    return _preprocess


def apply_pooling_attention_mask_patch() -> bool:
    """Patch vLLM v1 GPUModelRunner._preprocess. Returns True if applied or already patched."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.warning("vllm.v1.worker.gpu_model_runner not found; attention_mask patch skipped")
        return False

    if getattr(GPUModelRunner, _PATCH_ATTR, False):
        return True

    orig = GPUModelRunner._preprocess
    GPUModelRunner._preprocess = _make_patched_preprocess(orig)
    setattr(GPUModelRunner, _PATCH_ATTR, True)
    logger.info("Patched GPUModelRunner._preprocess for pooling attention_mask forwarding")
    return True
