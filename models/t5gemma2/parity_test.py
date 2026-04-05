#!/usr/bin/env python3
"""T5Gemma2 parity test for text and multimodal paths."""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch


def _init_vllm_env():
    """Initialize minimal vLLM distributed context for standalone tests."""

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    vllm_config = VllmConfig(compilation_config=CompilationConfig())
    ctx = set_current_vllm_config(vllm_config)
    ctx.__enter__()
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    return ctx


_CTX = _init_vllm_env()

try:
    from transformers import (
        AutoTokenizer,
    )
    from transformers import (
        T5Gemma2ForConditionalGeneration as HFT5Gemma2ForConditionalGeneration,
    )
except ImportError:
    AutoTokenizer = None
    HFT5Gemma2ForConditionalGeneration = None

from models.t5gemma2.t5gemma2_model import T5Gemma2ForConditionalGeneration  # noqa: E402

MODEL_NAME = "google/t5gemma-2-270m-270m"
TEXT_TEST_CASES = [
    (
        "Translate English to French: Hello world.",
        "Bonjour le monde.",
    ),
    (
        "Summarize: The quick brown fox jumps over the lazy dog.",
        "A fox jumps over a dog.",
    ),
]
MM_TARGETS = ["A synthetic image description."]


def _set_reference_path(enabled: bool) -> None:
    if enabled:
        os.environ["VLLM_FACTORY_T5GEMMA2_REFERENCE_PATH"] = "1"
    else:
        os.environ.pop("VLLM_FACTORY_T5GEMMA2_REFERENCE_PATH", None)


def _build_decoder_inputs(
    tokenizer,
    target_texts: list[str],
    model,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = tokenizer(
        target_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    bos_token_id = model.config.decoder.bos_token_id
    pad_token_id = model.config.decoder.pad_token_id

    decoder_input_ids = torch.full(
        (targets.input_ids.shape[0], targets.input_ids.shape[1] + 1),
        pad_token_id,
        dtype=torch.long,
    )
    decoder_input_ids[:, 0] = bos_token_id
    decoder_input_ids[:, 1:] = targets.input_ids
    decoder_attention_mask = decoder_input_ids.ne(pad_token_id)
    return decoder_input_ids, decoder_attention_mask


def _build_multimodal_inputs(tokenizer, config, device: torch.device) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(
        ["Describe the image in one sentence."],
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    placeholder_ids = torch.full(
        (1, config.encoder.mm_tokens_per_image),
        config.encoder.image_token_index,
        dtype=torch.long,
        device=device,
    )
    input_ids = torch.cat([prompt_ids, placeholder_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    image_size = config.encoder.vision_config.image_size
    pixel_values = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device=device,
        dtype=torch.float32,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }


def load_hf_model():
    if HFT5Gemma2ForConditionalGeneration is None or AutoTokenizer is None:
        raise RuntimeError(
            "This parity test requires a transformers build that includes "
            "`T5Gemma2ForConditionalGeneration`."
        )

    print(f"Loading HF model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = HFT5Gemma2ForConditionalGeneration.from_pretrained(MODEL_NAME)
    hf_model.eval()
    return hf_model, tokenizer


def load_our_model(hf_model, *, reference_path: bool):
    _set_reference_path(reference_path)
    label = "reference" if reference_path else "optimized"
    print(f"Loading vLLM Factory model ({label})")
    our_model = T5Gemma2ForConditionalGeneration(hf_model.config)
    our_model.eval()
    our_model.load_weights(hf_model.state_dict().items())
    return our_model


def compare_tensors(name: str, ref: torch.Tensor, actual: torch.Tensor) -> bool:
    abs_diff = (ref - actual).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    atol = 1e-4
    rtol = 1e-3
    ok = torch.allclose(ref, actual, atol=atol, rtol=rtol)
    print(
        f"  {name:<20} shape={tuple(ref.shape)} "
        f"max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} "
        f"status={'PASS' if ok else 'FAIL'}"
    )
    return ok


def _run_text_test(hf_model, our_model, tokenizer, device: torch.device) -> bool:
    source_texts = [source for source, _ in TEXT_TEST_CASES]
    target_texts = [target for _, target in TEXT_TEST_CASES]

    encoder_inputs = tokenizer(source_texts, return_tensors="pt", padding=True).to(device)
    decoder_input_ids, decoder_attention_mask = _build_decoder_inputs(
        tokenizer,
        target_texts,
        hf_model,
    )
    decoder_input_ids = decoder_input_ids.to(device)
    decoder_attention_mask = decoder_attention_mask.to(device)

    with torch.no_grad():
        hf_model_outputs = hf_model.model(
            input_ids=encoder_inputs.input_ids,
            attention_mask=encoder_inputs.attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        hf_lm_outputs = hf_model(
            input_ids=encoder_inputs.input_ids,
            attention_mask=encoder_inputs.attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        our_encoder_hidden = our_model.get_encoder_outputs(
            encoder_inputs.input_ids,
            attention_mask=encoder_inputs.attention_mask,
        )
        our_decoder_hidden = our_model(
            input_ids=encoder_inputs.input_ids,
            attention_mask=encoder_inputs.attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        our_logits = our_model.compute_logits(our_decoder_hidden)

    all_ok = True
    print("\nText parity")
    all_ok &= compare_tensors(
        "encoder_hidden",
        hf_model_outputs.encoder_last_hidden_state,
        our_encoder_hidden,
    )
    all_ok &= compare_tensors(
        "decoder_hidden",
        hf_model_outputs.last_hidden_state,
        our_decoder_hidden,
    )
    all_ok &= compare_tensors("logits", hf_lm_outputs.logits, our_logits)
    return all_ok


def _run_multimodal_test(hf_model, our_model, tokenizer, device: torch.device) -> bool:
    mm_inputs = _build_multimodal_inputs(tokenizer, hf_model.config, device)
    decoder_input_ids, decoder_attention_mask = _build_decoder_inputs(
        tokenizer,
        MM_TARGETS,
        hf_model,
    )
    decoder_input_ids = decoder_input_ids.to(device)
    decoder_attention_mask = decoder_attention_mask.to(device)

    with torch.no_grad():
        hf_model_outputs = hf_model.model(
            input_ids=mm_inputs["input_ids"],
            attention_mask=mm_inputs["attention_mask"],
            pixel_values=mm_inputs["pixel_values"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        hf_lm_outputs = hf_model(
            input_ids=mm_inputs["input_ids"],
            attention_mask=mm_inputs["attention_mask"],
            pixel_values=mm_inputs["pixel_values"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        our_encoder_hidden = our_model.get_encoder_outputs(
            mm_inputs["input_ids"],
            attention_mask=mm_inputs["attention_mask"],
            pixel_values=mm_inputs["pixel_values"],
        )
        our_decoder_hidden = our_model(
            input_ids=mm_inputs["input_ids"],
            attention_mask=mm_inputs["attention_mask"],
            pixel_values=mm_inputs["pixel_values"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        our_logits = our_model.compute_logits(our_decoder_hidden)

    all_ok = True
    print("\nMultimodal parity")
    all_ok &= compare_tensors(
        "mm_encoder_hidden",
        hf_model_outputs.encoder_last_hidden_state,
        our_encoder_hidden,
    )
    all_ok &= compare_tensors(
        "mm_decoder_hidden",
        hf_model_outputs.last_hidden_state,
        our_decoder_hidden,
    )
    all_ok &= compare_tensors("mm_logits", hf_lm_outputs.logits, our_logits)
    return all_ok


def run_test(hf_model, tokenizer, *, reference_path: bool) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_model = hf_model.cpu()
    our_model = load_our_model(hf_model, reference_path=reference_path)
    hf_model = hf_model.to(device)
    our_model = our_model.to(device)

    all_ok = True
    print("\n" + "=" * 72)
    print(f"T5Gemma2 PARITY TEST ({'reference' if reference_path else 'optimized'})")
    print("=" * 72)
    all_ok &= _run_text_test(hf_model, our_model, tokenizer, device)
    all_ok &= _run_multimodal_test(hf_model, our_model, tokenizer, device)
    print("=" * 72)
    return all_ok


if __name__ == "__main__":
    try:
        hf_model, tokenizer = load_hf_model()
        reference_ok = run_test(hf_model, tokenizer, reference_path=True)
        optimized_ok = run_test(hf_model, tokenizer, reference_path=False)
        sys.exit(0 if reference_ok and optimized_ok else 1)
    finally:
        _set_reference_path(False)
        if _CTX is not None:
            _CTX.__exit__(None, None, None)
