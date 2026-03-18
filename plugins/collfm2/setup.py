from setuptools import setup

setup(
    name="vllm-factory-collfm2",
    version="0.1.0",
    description="ColLFM2 plugin for vLLM — lightweight visual document retrieval",
    py_modules=[
        "collfm2.__init__",
    ],
    package_dir={"collfm2": "."},
    packages=["collfm2"],
    python_requires=">=3.11",
    install_requires=["vllm==0.15.1", "torch>=2.0", "transformers>=4.40"],
    entry_points={"vllm.general_plugins": ["collfm2 = collfm2:register"]},
)
