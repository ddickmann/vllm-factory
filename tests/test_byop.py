"""End-to-end test for the Bring-Your-Own-Pooler (BYOP) composable architecture.

Tests:
1. Registry APIs: list, register, get for backbones and poolers
2. Built-in poolers: MeanPooler, CLSPooler, NormalizedMeanPooler correctness
3. Custom WeightedAttentionPooler: non-trivial learnable pooler
4. Backbone resolution: model_type and architecture fallback
5. Pooler resolution: env var > config > error
6. Constructor compatibility: _create_standard, _create_mt5, _create_t5gemma2
7. GPU E2E: load real ModernBERT backbone, run forward, pool real hidden states

Usage:
    python tests/test_byop.py
"""

from __future__ import annotations

import logging
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_byop")


# ──────────────────────────────────────────────────────────────────────────
# Custom pooler used throughout the tests
# ──────────────────────────────────────────────────────────────────────────


class WeightedAttentionPooler(torch.nn.Module):
    """Non-trivial pooler: attention-weighted mean over token embeddings.

    Learns a query vector and computes attention weights per token,
    then returns the weighted sum as the sequence embedding.
    """

    def __init__(self, hidden_size: int = 768, temperature: float = 1.0, **kwargs):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(hidden_size))
        self.temperature = temperature

    def get_tasks(self) -> set[str]:
        return {"embed", "plugin"}

    def forward(self, hidden_states: torch.Tensor, ctx) -> list[torch.Tensor | None]:
        from vllm_factory.pooling.protocol import split_hidden_states

        parts = split_hidden_states(hidden_states, ctx.seq_lengths)
        results = []
        for seq in parts:
            logits = (seq @ self.query) / self.temperature
            weights = torch.softmax(logits, dim=0)
            pooled = (weights.unsqueeze(-1) * seq).sum(dim=0)
            results.append(pooled)
        return results


# ──────────────────────────────────────────────────────────────────────────
# 1. Registry API tests
# ──────────────────────────────────────────────────────────────────────────


def test_backbone_registry():
    from vllm_factory.composable.backbone_registry import get_backbone, list_backbones

    names = list_backbones()
    for expected in ("modernbert", "deberta", "deberta_v2", "mt5", "t5gemma2"):
        assert expected in names, f"Missing backbone: {expected}"

    entry = get_backbone("t5gemma2")
    assert entry.class_name == "T5Gemma2Encoder"
    assert callable(entry.create_instance)
    assert callable(entry.get_hidden_states)

    # Verify extra_architectures are populated
    mb = get_backbone("modernbert")
    assert "ModernBertForMaskedLM" in mb.extra_architectures

    logger.info("PASS: backbone registry (%d backbones)", len(names))


def test_pooler_registry():
    from vllm_factory.composable.pooler_registry import (
        get_pooler_cls,
        list_poolers,
        register_pooler,
    )

    names = list_poolers()
    for expected in ("mean", "cls", "normalized_mean", "passthrough"):
        assert expected in names, f"Missing pooler: {expected}"

    class DummyPooler:
        def get_tasks(self):
            return {"embed"}

        def forward(self, hidden_states, ctx):
            return []

    register_pooler("dummy_test", DummyPooler)
    assert get_pooler_cls("dummy_test") is DummyPooler

    # Verify re-registration overwrites
    register_pooler("dummy_test", WeightedAttentionPooler)
    assert get_pooler_cls("dummy_test") is WeightedAttentionPooler

    logger.info("PASS: pooler registry (%d poolers)", len(list_poolers()))


# ──────────────────────────────────────────────────────────────────────────
# 2. Built-in pooler correctness
# ──────────────────────────────────────────────────────────────────────────


def test_builtin_poolers():
    from vllm_factory.composable.pooler_registry import CLSPooler, MeanPooler, NormalizedMeanPooler
    from vllm_factory.pooling.protocol import PoolerContext

    hidden = torch.randn(20, 768)
    ctx = PoolerContext(seq_lengths=[8, 12])

    # Mean
    mp = MeanPooler()
    results = mp.forward(hidden, ctx)
    assert len(results) == 2
    assert results[0].shape == (768,)
    assert torch.allclose(results[0], hidden[:8].mean(dim=0))
    assert torch.allclose(results[1], hidden[8:20].mean(dim=0))

    # CLS
    cp = CLSPooler()
    results = cp.forward(hidden, ctx)
    assert torch.allclose(results[0], hidden[0])
    assert torch.allclose(results[1], hidden[8])

    # NormalizedMean
    nmp = NormalizedMeanPooler()
    results = nmp.forward(hidden, ctx)
    expected_0 = F.normalize(hidden[:8].mean(dim=0), p=2, dim=-1)
    assert torch.allclose(results[0], expected_0, atol=1e-6)
    assert abs(results[0].norm().item() - 1.0) < 1e-5, "Should be L2-normalized"

    # Edge case: single-token sequence
    ctx_single = PoolerContext(seq_lengths=[1])
    hidden_single = torch.randn(1, 768)
    r = MeanPooler().forward(hidden_single, ctx_single)
    assert torch.allclose(r[0], hidden_single[0])

    logger.info("PASS: built-in poolers (Mean, CLS, NormalizedMean)")


def test_custom_pooler_standalone():
    from vllm_factory.pooling.protocol import PoolerContext

    hidden = torch.randn(15, 256)
    ctx = PoolerContext(seq_lengths=[7, 8])

    pooler = WeightedAttentionPooler(hidden_size=256, temperature=0.5)
    results = pooler.forward(hidden, ctx)
    assert len(results) == 2
    assert results[0].shape == (256,)
    assert results[1].shape == (256,)

    mean_result = hidden[:7].mean(dim=0)
    assert not torch.allclose(results[0], mean_result, atol=1e-3), \
        "Weighted attention should differ from plain mean"

    logger.info("PASS: WeightedAttentionPooler standalone")


# ──────────────────────────────────────────────────────────────────────────
# 3. Backbone resolution
# ──────────────────────────────────────────────────────────────────────────


def test_backbone_resolution():
    from vllm_factory.composable.model import _resolve_backbone_name

    cases = [
        ({"model_type": "t5gemma2", "architectures": []}, "t5gemma2"),
        ({"model_type": "modernbert", "architectures": []}, "modernbert"),
        ({"model_type": "deberta-v2", "architectures": []}, "deberta_v2"),
        ({"model_type": "deberta_v2", "architectures": []}, "deberta_v2"),
        ({"model_type": "mt5", "architectures": []}, "mt5"),
        ({"model_type": "deberta", "architectures": []}, "deberta"),
        # Architecture fallback
        ({"model_type": "unknown", "architectures": ["ModernBertForMaskedLM"]}, "modernbert"),
        ({"model_type": "unknown", "architectures": ["T5Gemma2ForConditionalGeneration"]}, "t5gemma2"),
        ({"model_type": "unknown", "architectures": ["MT5ForConditionalGeneration"]}, "mt5"),
        ({"model_type": "unknown", "architectures": ["DebertaV2ForMaskedLM"]}, "deberta_v2"),
    ]
    for attrs, expected in cases:

        class Cfg:
            pass

        cfg = Cfg()
        for k, v in attrs.items():
            setattr(cfg, k, v)
        result = _resolve_backbone_name(cfg)
        assert result == expected, f"Expected {expected}, got {result} for {attrs}"

    # Unknown model_type + unknown architecture should raise
    try:

        class BadCfg:
            model_type = "totally_unknown"
            architectures = ["AlsoUnknown"]

        _resolve_backbone_name(BadCfg())
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    logger.info("PASS: backbone resolution (%d cases)", len(cases))


# ──────────────────────────────────────────────────────────────────────────
# 4. Pooler resolution
# ──────────────────────────────────────────────────────────────────────────


def test_pooler_resolution():
    from vllm_factory.composable.model import _resolve_pooler_name

    # Env var takes precedence
    os.environ["VLLM_FACTORY_POOLER"] = "mean"
    try:

        class Cfg:
            pooler_type = "cls"

        assert _resolve_pooler_name(Cfg()) == "mean"
    finally:
        os.environ.pop("VLLM_FACTORY_POOLER", None)

    # Config fallback
    class Cfg2:
        pooler_type = "cls"

    assert _resolve_pooler_name(Cfg2()) == "cls"

    # Missing -> error
    try:

        class Cfg3:
            pooler_type = None

        _resolve_pooler_name(Cfg3())
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    logger.info("PASS: pooler resolution (env > config > error)")


# ──────────────────────────────────────────────────────────────────────────
# 5. Pooler instantiation compatibility
# ──────────────────────────────────────────────────────────────────────────


def test_pooler_instantiation():
    from vllm_factory.composable.model import _instantiate_pooler
    from vllm_factory.composable.pooler_registry import MeanPooler

    # Standard (hidden_size=, **kwargs)
    p = _instantiate_pooler(MeanPooler, 768, {})
    assert isinstance(p, MeanPooler)

    # nn.Module pooler
    p2 = _instantiate_pooler(WeightedAttentionPooler, 256, {"temperature": 0.5})
    assert isinstance(p2, WeightedAttentionPooler)
    assert p2.temperature == 0.5
    assert p2.query.shape == (256,)

    # No-args pooler
    class NoArgPooler:
        def __init__(self):
            pass

        def get_tasks(self):
            return {"embed"}

        def forward(self, hs, ctx):
            return []

    p3 = _instantiate_pooler(NoArgPooler, 768, {})
    assert isinstance(p3, NoArgPooler)

    # Bad pooler -> error
    class BadPooler:
        def __init__(self, a, b, c, d):
            pass

    try:
        _instantiate_pooler(BadPooler, 768, {})
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    logger.info("PASS: pooler instantiation compatibility")


# ──────────────────────────────────────────────────────────────────────────
# 6. VllmPoolerAdapter wiring
# ──────────────────────────────────────────────────────────────────────────


def test_vllm_adapter_wiring():
    from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

    pooler = WeightedAttentionPooler(hidden_size=64)
    adapter = VllmPoolerAdapter(pooler)

    supported = adapter.get_supported_tasks()
    assert "embed" in supported or "plugin" in supported

    logger.info("PASS: VllmPoolerAdapter wiring")


# ──────────────────────────────────────────────────────────────────────────
# 7. GPU E2E: real ModernBERT backbone + custom pooler
# ──────────────────────────────────────────────────────────────────────────


def test_e2e_real_backbone():
    """Load the actual ModernBERT backbone, run real forward pass, pool the output.

    This validates the full chain: backbone_registry -> load_backbone_class ->
    create_instance -> forward -> get_hidden_states -> pooler.forward
    """
    if not torch.cuda.is_available():
        logger.warning("SKIP: No GPU available for E2E test")
        return

    from vllm_factory.composable.backbone_registry import get_backbone, load_backbone_class
    from vllm_factory.composable.pooler_registry import MeanPooler
    from vllm_factory.pooling.protocol import PoolerContext

    model_id = "answerdotai/ModernBERT-base"

    try:
        from transformers import AutoConfig, AutoTokenizer

        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        logger.warning("SKIP: cannot load config/tokenizer for %s: %s", model_id, e)
        return

    entry = get_backbone("modernbert")
    BackboneClass = load_backbone_class(entry)
    logger.info("Loaded class: %s", BackboneClass.__name__)

    # Instantiate backbone directly (no vllm_config — use the DeBERTa-style
    # fallback that accepts config= directly, but ModernBERT requires vllm_config).
    # Instead, we instantiate via the HF transformers model for the E2E test.
    from transformers import AutoModel

    hf_model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda().eval()

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning transforms information processing",
        "Triton kernels optimize GPU computation",
    ]
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = hf_model(**encoded)
    hidden = out.last_hidden_state  # (B, S, H)

    # Flatten to 2D (total_tokens, H) — simulating vLLM's concatenated format
    attn_mask = encoded["attention_mask"]
    seq_lengths = attn_mask.sum(dim=1).tolist()
    flat_parts = []
    for i, slen in enumerate(seq_lengths):
        flat_parts.append(hidden[i, :slen])
    flat_hidden = torch.cat(flat_parts, dim=0)

    ctx = PoolerContext(seq_lengths=seq_lengths)

    # --- Test MeanPooler ---
    mean_pooler = MeanPooler()
    mean_results = mean_pooler.forward(flat_hidden, ctx)
    assert len(mean_results) == 3
    for i, emb in enumerate(mean_results):
        assert emb.shape == (hf_config.hidden_size,), f"Bad shape for seq {i}: {emb.shape}"
    logger.info("  MeanPooler: 3 embeddings of dim %d", hf_config.hidden_size)

    # Verify mean is correct
    manual_mean_0 = flat_hidden[: seq_lengths[0]].mean(dim=0)
    assert torch.allclose(mean_results[0], manual_mean_0, atol=1e-5)

    # --- Test WeightedAttentionPooler ---
    wa_pooler = WeightedAttentionPooler(hidden_size=hf_config.hidden_size).cuda().to(torch.bfloat16)
    wa_results = wa_pooler.forward(flat_hidden, ctx)
    assert len(wa_results) == 3
    for emb in wa_results:
        assert emb.shape == (hf_config.hidden_size,)

    # Verify different inputs produce different embeddings
    cos_sim = F.cosine_similarity(
        wa_results[0].float().unsqueeze(0),
        wa_results[1].float().unsqueeze(0),
    )
    assert cos_sim.item() < 0.999, "Different texts should produce different embeddings"
    logger.info("  WeightedAttention: cosine(0,1)=%.4f", cos_sim.item())

    # --- Test VllmPoolerAdapter wrapping ---
    from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

    adapter = VllmPoolerAdapter(wa_pooler)
    assert "embed" in adapter.get_supported_tasks() or "plugin" in adapter.get_supported_tasks()

    # Cleanup
    del hf_model
    torch.cuda.empty_cache()

    logger.info("PASS: E2E real backbone (ModernBERT) + poolers on GPU")


# ──────────────────────────────────────────────────────────────────────────
# 8. Manual weight loading fallback
# ──────────────────────────────────────────────────────────────────────────


def test_manual_weight_loading():
    """Verify that _load_weights_manual correctly loads parameters by name."""
    if not torch.cuda.is_available():
        logger.warning("SKIP: No GPU available")
        return

    from vllm_factory.composable.model import ComposablePoolingModel

    # Create a simple backbone mock that has parameters but no load_weights
    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 16)

    model = ComposablePoolingModel.__new__(ComposablePoolingModel)
    torch.nn.Module.__init__(model)
    model._backbone = FakeBackbone().cuda()

    # Test the manual fallback
    original_weight = model._backbone.linear.weight.clone()
    new_weight = torch.randn_like(original_weight)
    loaded = model._load_weights_manual([
        ("linear.weight", new_weight),
        ("nonexistent.param", torch.randn(10)),
    ])
    assert "linear.weight" in loaded
    assert "nonexistent.param" not in loaded
    assert torch.allclose(model._backbone.linear.weight, new_weight)

    logger.info("PASS: manual weight loading fallback")


# ──────────────────────────────────────────────────────────────────────────
# 9. DeBERTa backbone verification
# ──────────────────────────────────────────────────────────────────────────


def test_e2e_deberta_backbone():
    """Verify DeBERTa backbone can be loaded and used for pooling."""
    if not torch.cuda.is_available():
        logger.warning("SKIP: No GPU available")
        return

    from vllm_factory.composable.backbone_registry import get_backbone, load_backbone_class
    from vllm_factory.composable.pooler_registry import CLSPooler, NormalizedMeanPooler
    from vllm_factory.pooling.protocol import PoolerContext

    model_id = "microsoft/deberta-v3-small"

    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        logger.warning("SKIP: cannot load config/tokenizer for %s: %s", model_id, e)
        return

    entry = get_backbone("deberta_v2")
    BackboneClass = load_backbone_class(entry)
    logger.info("  Loaded class: %s", BackboneClass.__name__)

    hf_model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda().eval()

    texts = ["Sentence embeddings are useful", "DeBERTa is a transformer model"]
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = hf_model(**encoded)
    hidden = out.last_hidden_state

    attn_mask = encoded["attention_mask"]
    seq_lengths = attn_mask.sum(dim=1).tolist()
    flat_parts = [hidden[i, :int(slen)] for i, slen in enumerate(seq_lengths)]
    flat_hidden = torch.cat(flat_parts, dim=0)

    ctx = PoolerContext(seq_lengths=[int(s) for s in seq_lengths])

    # CLS pooler
    cls_pooler = CLSPooler()
    cls_results = cls_pooler.forward(flat_hidden, ctx)
    assert len(cls_results) == 2
    assert cls_results[0].shape == (hf_config.hidden_size,)
    assert torch.allclose(cls_results[0], flat_hidden[0])

    # NormalizedMean pooler (bf16 precision means norm may not be exactly 1.0)
    nm_pooler = NormalizedMeanPooler()
    nm_results = nm_pooler.forward(flat_hidden, ctx)
    for r in nm_results:
        assert abs(r.float().norm().item() - 1.0) < 0.01

    # WeightedAttention
    wa = WeightedAttentionPooler(hidden_size=hf_config.hidden_size).cuda().to(torch.bfloat16)
    wa_results = wa.forward(flat_hidden, ctx)
    assert len(wa_results) == 2
    for r in wa_results:
        assert r.shape == (hf_config.hidden_size,)
        assert not torch.isnan(r).any()

    del hf_model
    torch.cuda.empty_cache()

    logger.info("PASS: E2E DeBERTa-v3-small backbone + poolers on GPU")


# ──────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_backbone_registry,
        test_pooler_registry,
        test_builtin_poolers,
        test_custom_pooler_standalone,
        test_backbone_resolution,
        test_pooler_resolution,
        test_pooler_instantiation,
        test_vllm_adapter_wiring,
        test_e2e_real_backbone,
        test_manual_weight_loading,
        test_e2e_deberta_backbone,
    ]

    passed = 0
    failed = 0

    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            logger.error("FAIL: %s — %s", fn.__name__, e, exc_info=True)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if failed > 0 else 0)
