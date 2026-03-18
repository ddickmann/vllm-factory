"""mmBERT-GLiNER vLLM Plugin — pip install -e ."""

from setuptools import find_packages, setup

setup(
    name="vllm-factory-mmbert-gliner",
    version="0.1.0",
    description="GLiNER NER plugin for vLLM — custom ModernBERT + span extraction",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "vllm==0.15.1",
        "torch>=2.0",
        "transformers>=4.40",
        "triton>=2.0",
    ],
    entry_points={
        "vllm.general_plugins": [
            "mmbert_gliner = mmbert_gliner:register",
        ],
    },
)
