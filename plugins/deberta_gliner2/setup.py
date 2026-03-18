"""Setup for GLiNER2 vLLM Plugin."""

from setuptools import setup

setup(
    name="vllm-factory-deberta-gliner2",
    version="0.1.0",
    description="GLiNER2 plugin for vLLM — schema-based multi-task extraction",
    package_dir={"deberta_gliner2": "."},
    packages=["deberta_gliner2"],
    python_requires=">=3.11",
    install_requires=[
        "vllm==0.15.1",
        "torch>=2.0",
        "transformers>=4.40",
    ],
    entry_points={
        "vllm.general_plugins": [
            "deberta_gliner2 = deberta_gliner2:register",
        ],
    },
)
