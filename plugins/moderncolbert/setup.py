"""
ModernColBERT vLLM Plugin — pip install -e .

Registers ModernBertForColBERT with vLLM's model registry so it can be served with:
    vllm serve <model_id> --trust-remote-code
"""

from setuptools import setup

setup(
    name="vllm-factory-moderncolbert",
    version="0.1.0",
    description="ModernColBERT plugin for vLLM — multi-vector retrieval with custom ModernBERT",
    package_dir={"moderncolbert": "."},
    packages=["moderncolbert"],
    python_requires=">=3.11",
    install_requires=[
        "vllm==0.15.1",
        "torch>=2.0",
        "transformers>=4.40",
        "triton>=2.0",
        "safetensors",
    ],
    entry_points={
        "vllm.general_plugins": [
            "moderncolbert = moderncolbert:register",
        ],
    },
)
