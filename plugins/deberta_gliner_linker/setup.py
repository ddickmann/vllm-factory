"""Setup for GLiNER-Linker vLLM Plugin."""

from setuptools import setup

setup(
    name="vllm-factory-deberta-gliner-linker",
    version="0.1.0",
    description="GLiNER-Linker plugin for vLLM — DeBERTa v1 bi-encoder entity linking",
    package_dir={"deberta_gliner_linker": "."},
    packages=["deberta_gliner_linker"],
    python_requires=">=3.11",
    install_requires=[
        "vllm==0.15.1",
        "torch>=2.0",
        "transformers>=4.40",
        "huggingface-hub",
    ],
    entry_points={
        "vllm.general_plugins": [
            "deberta_gliner_linker = deberta_gliner_linker:register",
        ],
    },
)
