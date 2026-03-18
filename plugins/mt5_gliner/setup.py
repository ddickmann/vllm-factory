"""Setup for mT5 GLiNER vLLM Plugin."""

from setuptools import setup

setup(
    name="vllm-factory-mt5-gliner",
    version="0.1.0",
    description="mT5 GLiNER plugin for vLLM — multilingual zero-shot NER",
    package_dir={"mt5_gliner": "."},
    packages=["mt5_gliner"],
    python_requires=">=3.11",
    install_requires=[
        "vllm==0.15.1",
        "torch>=2.0",
        "transformers>=4.40",
    ],
    entry_points={
        "vllm.general_plugins": [
            "mt5_gliner = mt5_gliner:register",
        ],
    },
)
