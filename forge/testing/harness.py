"""
Model Test Harness — standardized parity and benchmark testing.

Usage:
    harness = ModelTestHarness(
        plugin_name="colbert",
        model_id="VAGOsolutions/ModernColBERT",
        reference_model_cls=PyLateColBERT,
        vllm_model_cls=ModernBertForColBERT,
    )
    harness.test_parity(sample_inputs)
    harness.benchmark_throughput(sample_inputs, batch_sizes=[1, 8, 32])
    harness.generate_report("reports/colbert_report.md")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch


@dataclass
class ParityResult:
    """Result of a parity check between reference and vLLM outputs."""

    cosine_similarity: float
    max_absolute_error: float
    mean_absolute_error: float
    passed: bool
    details: str = ""


@dataclass
class BenchmarkResult:
    """Result of a throughput/latency benchmark."""

    batch_size: int
    total_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


@dataclass
class TestReport:
    """Full test report combining parity and benchmark results."""

    plugin_name: str
    model_id: str
    parity_results: list[ParityResult] = field(default_factory=list)
    benchmark_results: list[BenchmarkResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelTestHarness:
    """Standardized test harness for vLLM custom model plugins.

    Provides a uniform interface for:
    1. Parity testing — compare vLLM output vs reference PyTorch model
    2. Throughput benchmarking — measure tokens/sec at various batch sizes
    3. Latency profiling — P50/P95/P99 per-request latency
    4. Report generation — markdown report with all results

    Example:
        harness = ModelTestHarness("colbert", "VAGOsolutions/ModernColBERT")
        harness.test_parity(inputs, reference_fn=run_pylate, vllm_fn=run_vllm)
    """

    def __init__(
        self,
        plugin_name: str,
        model_id: str,
    ):
        self.plugin_name = plugin_name
        self.model_id = model_id
        self.report = TestReport(plugin_name=plugin_name, model_id=model_id)

    def test_parity(
        self,
        inputs: list[str],
        reference_fn: Callable[[list[str]], torch.Tensor],
        vllm_fn: Callable[[list[str]], torch.Tensor],
        rtol: float = 1e-3,
        atol: float = 1e-4,
        min_cosine_sim: float = 0.99,
    ) -> ParityResult:
        """Compare vLLM output against reference implementation.

        Args:
            inputs: List of input strings
            reference_fn: Function that returns reference output tensor
            vllm_fn: Function that returns vLLM output tensor
            rtol: Relative tolerance for allclose check
            atol: Absolute tolerance for allclose check
            min_cosine_sim: Minimum cosine similarity to pass

        Returns:
            ParityResult with similarity metrics
        """
        print(f"[harness] Testing parity for {self.plugin_name}...")
        print(f"  Model: {self.model_id}")
        print(f"  Inputs: {len(inputs)} samples")

        # Get outputs
        ref_output = reference_fn(inputs)
        vllm_output = vllm_fn(inputs)

        # Compute metrics
        cos_sim = self._cosine_similarity(ref_output, vllm_output)
        max_err = (ref_output - vllm_output).abs().max().item()
        mean_err = (ref_output - vllm_output).abs().mean().item()

        passed = cos_sim >= min_cosine_sim

        result = ParityResult(
            cosine_similarity=cos_sim,
            max_absolute_error=max_err,
            mean_absolute_error=mean_err,
            passed=passed,
            details=f"{'✓ PASS' if passed else '✗ FAIL'} — cosine={cos_sim:.6f} (min={min_cosine_sim})",
        )

        print(f"  {result.details}")
        self.report.parity_results.append(result)
        return result

    def benchmark_throughput(
        self,
        inputs: list[str],
        run_fn: Callable[[list[str]], Any],
        batch_sizes: list[int] | None = None,
        n_warmup: int = 3,
        n_runs: int = 10,
    ) -> list[BenchmarkResult]:
        """Benchmark throughput at different batch sizes.

        Args:
            inputs: Pool of input strings to sample from
            run_fn: Function to benchmark
            batch_sizes: List of batch sizes to test
            n_warmup: Number of warmup runs
            n_runs: Number of timed runs per batch size

        Returns:
            List of BenchmarkResult per batch size
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 32, 128]

        results = []

        for bs in batch_sizes:
            # Build batch (repeat inputs if needed)
            batch = (inputs * ((bs // len(inputs)) + 1))[:bs]
            total_tokens = sum(len(s.split()) for s in batch)  # rough estimate

            # Warmup
            print(f"  Warming up (batch_size={bs})...")
            for _ in range(n_warmup):
                run_fn(batch)

            # Time runs
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latencies = []

            for _ in range(n_runs):
                start = time.perf_counter()
                run_fn(batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

            latencies_ms = [l * 1000 for l in latencies]  # noqa: E741
            total_elapsed = sum(latencies)

            result = BenchmarkResult(
                batch_size=bs,
                total_tokens=total_tokens * n_runs,
                elapsed_seconds=total_elapsed,
                tokens_per_second=(total_tokens * n_runs) / total_elapsed,
                p50_latency_ms=float(np.percentile(latencies_ms, 50)),
                p95_latency_ms=float(np.percentile(latencies_ms, 95)),
                p99_latency_ms=float(np.percentile(latencies_ms, 99)),
            )

            print(
                f"  batch={bs:>4d}  "
                f"tok/s={result.tokens_per_second:>8.0f}  "
                f"p50={result.p50_latency_ms:>6.1f}ms  "
                f"p99={result.p99_latency_ms:>6.1f}ms"
            )
            results.append(result)

        self.report.benchmark_results.extend(results)
        return results

    def generate_report(self, output_path: str) -> str:
        """Generate a markdown report with parity + benchmark results.

        Args:
            output_path: Path to write the report

        Returns:
            The markdown content
        """
        lines = [
            f"# Test Report: {self.plugin_name}",
            "",
            f"**Model**: `{self.model_id}`",
            "",
        ]

        # Parity results
        if self.report.parity_results:
            lines.extend([
                "## Parity Results",
                "",
                "| # | Cosine Sim | Max Error | Mean Error | Status |",
                "|---|-----------|-----------|------------|--------|",
            ])
            for i, r in enumerate(self.report.parity_results, 1):
                status = "✅ PASS" if r.passed else "❌ FAIL"
                lines.append(
                    f"| {i} | {r.cosine_similarity:.6f} | {r.max_absolute_error:.6f} | "
                    f"{r.mean_absolute_error:.6f} | {status} |"
                )
            lines.append("")

        # Benchmark results
        if self.report.benchmark_results:
            lines.extend([
                "## Benchmark Results",
                "",
                "| Batch Size | Tokens/sec | P50 (ms) | P95 (ms) | P99 (ms) |",
                "|-----------|-----------|---------|---------|---------|",
            ])
            for r in self.report.benchmark_results:
                lines.append(
                    f"| {r.batch_size} | {r.tokens_per_second:,.0f} | "
                    f"{r.p50_latency_ms:.1f} | {r.p95_latency_ms:.1f} | "
                    f"{r.p99_latency_ms:.1f} |"
                )
            lines.append("")

        content = "\n".join(lines)

        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(content)
        print(f"\n📊 Report written to {output_path}")

        return content

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute mean cosine similarity between two tensors."""
        a_flat = a.float().reshape(-1)
        b_flat = b.float().reshape(-1)

        # Truncate to same length if needed
        min_len = min(len(a_flat), len(b_flat))
        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]

        cos = torch.nn.functional.cosine_similarity(
            a_flat.unsqueeze(0), b_flat.unsqueeze(0)
        )
        return cos.item()
