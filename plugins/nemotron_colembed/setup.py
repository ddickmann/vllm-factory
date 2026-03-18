from setuptools import setup

setup(
    name="vllm-factory-nemotron-colembed",
    version="0.1.0",
    description="Nemotron ColEmbed plugin for vLLM — visual document retrieval with bidirectional Qwen3-VL",
    package_dir={"nemotron_colembed": "."},
    packages=["nemotron_colembed"],
    python_requires=">=3.11",
    install_requires=["vllm==0.15.1", "torch>=2.0", "transformers>=4.57.2"],
    entry_points={
        "vllm.general_plugins": [
            "nemotron_colembed = nemotron_colembed:register",
        ],
    },
)
