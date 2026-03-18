# Benchmark Reproducibility Protocol

This protocol standardizes benchmark runs so published throughput claims can be reproduced.

## 1) Pin environment

Use:

- `vllm==0.15.x`
- Python 3.11
- The dependency baseline in `benchmarks/requirements-reproducible.txt`

## 2) Capture environment metadata

Before each benchmark run:

```bash
python benchmarks/environment_capture.py --output benchmarks/results/environment.json
```

This captures Python, platform, torch, vLLM, and visible GPU metadata.

## 3) Benchmark run rules

- Warmup first (minimum 10 warmup iterations).
- Use fixed input corpus.
- Log batch sizes and concurrency levels.
- Record p50/p95/p99 latency and req/s.

## 4) Output schema

Store benchmark outputs using `benchmarks/report_schema.json`.

Minimum required fields:

- `benchmark_name`
- `model`
- `commit`
- `environment`
- `metrics`

## 5) Publication checklist

- Include hardware details.
- Include exact command used.
- Link the commit SHA and report artifact.
- State whether pooling patch was applied.
