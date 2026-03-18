from setuptools import setup

setup(
    name="vllm-factory-lfm2-colbert",
    version="0.1.0",
    description="LFM2-ColBERT plugin for vLLM — LFM2 encoder with ColBERT multi-vector pooler",
    package_dir={"lfm2_colbert": "."},
    packages=["lfm2_colbert"],
    python_requires=">=3.11",
    install_requires=["vllm==0.15.1", "torch>=2.0", "transformers>=4.40"],
    entry_points={"vllm.general_plugins": ["lfm2_colbert = lfm2_colbert:register"]},
)
