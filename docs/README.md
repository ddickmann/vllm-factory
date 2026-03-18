# Documentation

## Getting Started

- **[Quickstart](quickstart.md)** — install, serve, and test in 5 minutes
- **[Server Deployment Guide](server_guide.md)** — production config, Docker, multi-model, request formats
- **[Support Matrix](support_matrix.md)** — Python/vLLM/GPU compatibility and per-plugin notes

## Plugin Development

- **[Building a Plugin](building_a_plugin.md)** — step-by-step guide for new encoder plugins
- **[Plugin Integration Guide](PLUGIN_GUIDE.md)** — detailed 8-file plugin structure reference
- **[Architecture](architecture.md)** — how plugins load and integrate with vLLM

## GLiNER / Entity Linking

- **[gliner/README.md](gliner/README.md)** — status and serving guide for all GLiNER plugins
- **[gliner/INTEGRATION_GUIDE.md](gliner/INTEGRATION_GUIDE.md)** — full handover: patches, quirks, verification

## Reference

- **[Pooling Patch](pooling_patch.md)** — `extra_kwargs` passthrough for vLLM 0.15.x
- **[IO Processor Audit](io_processor_audit_matrix.md)** — IOProcessor migration audit for all plugins
- **[Benchmark Reproducibility](benchmark_reproducibility.md)** — protocol for reproducible benchmarks
- **[Release Process](release_process.md)** — versioning and CI policy
- **[macOS Setup](macos_vllm.md)** — local macOS development notes
