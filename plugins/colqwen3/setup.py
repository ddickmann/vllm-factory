from setuptools import setup

setup(
    name="vllm-factory-colqwen3",
    version="0.1.0",
    description="ColQwen3 plugin for vLLM — visual document retrieval with Qwen3-VL",
    package_dir={"colqwen3": "."},
    packages=["colqwen3"],
    python_requires=">=3.11",
    install_requires=["vllm==0.15.1", "torch>=2.0", "transformers>=4.40"],
    entry_points={"vllm.general_plugins": ["colqwen3 = colqwen3:register"]},
)
