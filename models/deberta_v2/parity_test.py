#!/usr/bin/env python3
"""
DeBERTa v2/v3 Parity Test — Hidden State Comparison.

Loads microsoft/deberta-v3-base with both HuggingFace and our vLLM encoder,
then compares the hidden state outputs to verify numerical parity.

Usage:
    cd /workspace/vllm-factory
    python models/deberta_v2/parity_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch


# ── vLLM environment bootstrap ──────────────────────────────────────
def _init_vllm_env():
    """Initialize minimal vLLM distributed + config context for standalone tests."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    vllm_config = VllmConfig(compilation_config=CompilationConfig(level=0))
    ctx = set_current_vllm_config(vllm_config)
    ctx.__enter__()
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="env://")
    ensure_model_parallel_initialized(tensor_model_parallel_size=1,
                                       pipeline_model_parallel_size=1)
    return ctx

# ── Bootstrap BEFORE any model imports ──────────────────────────────
_ctx = _init_vllm_env()

from transformers import AutoModel, AutoTokenizer  # noqa: E402

MODEL_NAME = "microsoft/deberta-v3-base"

TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "DeBERTa v3 uses disentangled attention with log-bucket relative positions.",
    "This is a test sentence for verifying parity between HuggingFace and vLLM implementations.",
]


def load_hf_model():
    """Load the reference HuggingFace DeBERTa v3 model."""
    print(f"Loading HF model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def load_vllm_model(hf_model):
    """Create our vLLM encoder and load weights from the HF model."""
    from models.deberta_v2.deberta_v2_encoder import DebertaV2EncoderModel

    config = hf_model.config
    print(f"\nConfig: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}")
    print(f"  relative_attention={getattr(config, 'relative_attention', False)}")
    print(f"  pos_att_type={getattr(config, 'pos_att_type', [])}")
    print(f"  max_relative_positions={getattr(config, 'max_relative_positions', -1)}")
    print(f"  position_buckets={getattr(config, 'position_buckets', -1)}")
    print(f"  share_att_key={getattr(config, 'share_att_key', False)}")
    print(f"  norm_rel_ebd={getattr(config, 'norm_rel_ebd', 'none')}")
    print(f"  conv_kernel_size={getattr(config, 'conv_kernel_size', 0)}")

    vllm_model = DebertaV2EncoderModel(config=config)
    vllm_model.eval()

    hf_state_dict = hf_model.state_dict()
    our_params = dict(vllm_model.named_parameters())
    our_buffers = dict(vllm_model.named_buffers())

    loaded = 0
    skipped = []

    for hf_name, hf_weight in hf_state_dict.items():
        our_name = hf_name
        if our_name.startswith("deberta."):
            our_name = our_name[len("deberta."):]
        our_name = our_name.replace(".attention.self.", ".attention.self_attn.")

        if our_name in our_params:
            param = our_params[our_name]
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, hf_weight)
                loaded += 1
            elif param.shape == hf_weight.shape:
                param.data.copy_(hf_weight)
                loaded += 1
            else:
                skipped.append(f"{our_name}: shape mismatch {param.shape} vs {hf_weight.shape}")
        elif our_name in our_buffers:
            our_buffers[our_name].copy_(hf_weight)
            loaded += 1
        else:
            skipped.append(f"{our_name} (from {hf_name}): not found")

    print(f"\nWeight loading: {loaded} loaded, {len(skipped)} skipped")
    if skipped:
        for s in skipped[:30]:
            print(f"  SKIP: {s}")
        if len(skipped) > 30:
            print(f"  ... and {len(skipped) - 30} more")

    return vllm_model


def compare_outputs(hf_model, vllm_model, tokenizer):
    """Compare hidden state outputs between HF and vLLM models."""
    print("\n" + "=" * 70)
    print("PARITY TEST: Comparing Hidden States")
    print("=" * 70)

    device = next(hf_model.parameters()).device
    vllm_model = vllm_model.to(device)

    all_pass = True

    for i, text in enumerate(TEST_TEXTS):
        print(f"\n--- Test {i+1}: \"{text[:60]}{'...' if len(text) > 60 else ''}\" ---")

        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs.get("token_type_ids")

        with torch.no_grad():
            hf_output = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            hf_hidden = hf_output.last_hidden_state

            vllm_hidden = vllm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        abs_diff = (hf_hidden - vllm_hidden).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        atol = 1e-4
        rtol = 1e-4
        close = torch.allclose(hf_hidden, vllm_hidden, atol=atol, rtol=rtol)
        within_tol = (abs_diff < atol).float().mean().item() * 100

        status = "✅ PASS" if close else "❌ FAIL"
        print(f"  Shape: {hf_hidden.shape}")
        print(f"  Max abs diff:  {max_diff:.6e}")
        print(f"  Mean abs diff: {mean_diff:.6e}")
        print(f"  Within atol={atol}: {within_tol:.2f}%")
        print(f"  Status: {status}")

        if not close:
            all_pass = False
            flat_diff = abs_diff.flatten()
            top_k = min(5, flat_diff.numel())
            top_vals, _ = flat_diff.topk(top_k)
            print(f"  Top {top_k} diffs: {[f'{v:.6e}' for v in top_vals.tolist()]}")

    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL TESTS PASSED — Hidden states match within tolerance!")
    else:
        print("❌ SOME TESTS FAILED — See details above")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    try:
        hf_model, tokenizer = load_hf_model()
        vllm_model = load_vllm_model(hf_model)
        success = compare_outputs(hf_model, vllm_model, tokenizer)
        sys.exit(0 if success else 1)
    finally:
        if _ctx is not None:
            _ctx.__exit__(None, None, None)
