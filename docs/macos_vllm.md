# macOS vLLM Setup Notes

This project supports macOS for local development and smoke checks. Production
throughput validation should run on Linux + NVIDIA GPUs.

## Canonical Local Setup (CPU/macOS-friendly)

```bash
python -m venv latence_venv
source latence_venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install "git+https://github.com/vllm-project/vllm.git@v0.15.1"
```

## Verify

```bash
python -c "import importlib.metadata as m; print(m.version('vllm'))"
vllm --version
```

Expected: `0.15.1`

## Important Caveats

- macOS setups may log warnings about GPU-only features (for example Triton/CUDA).
- Some compiled extension symbols can vary by local toolchain and torch versions.
- This is acceptable for plugin development, linting, and CPU-safe tests.
