from setuptools import setup, find_packages

setup(
    name="tt_xla",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.11",
        "flax>=0.7.2",
        "numpy>=1.24",
        "transformers",
        "datasets",
        "safetensors",
        "tqdm",
        "einops",
        "fsspec",
        "jaxtyping",
        "sentencepiece",
    ],
) 