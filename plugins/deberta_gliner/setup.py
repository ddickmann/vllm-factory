"""Setup for DeBERTa v2 GLiNER vLLM Plugin."""

from setuptools import setup

setup(
    name="vllm-factory-deberta-gliner",
    version="0.1.0",
    description="DeBERTa v2 GLiNER plugin for vLLM — zero-shot NER",
    package_dir={"deberta_gliner": "."},
    packages=["deberta_gliner"],
    python_requires=">=3.11",
    install_requires=[
        "vllm==0.15.1",
        "torch>=2.0",
        "transformers>=4.40",
    ],
    entry_points={
        "vllm.general_plugins": [
            "deberta_gliner = deberta_gliner:register",
        ],
    },
)
