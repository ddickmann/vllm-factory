from setuptools import find_packages, setup

setup(
    name="vllm-factory-embeddinggemma",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "vllm==0.15.1",
        "torch>=2.0",
        "transformers>=4.40",
    ],
    entry_points={
        "vllm.general_plugins": [
            "embeddinggemma = embeddinggemma:register",
        ],
    },
)
