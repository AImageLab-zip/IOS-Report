from setuptools import setup, find_packages

setup(
    name="pointqwen",
    version="0.1.0",
    description="Adapting Qwen3-VL for 3D Point Cloud Understanding",
    author="IOS-DraftReport Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "datasets>=2.14.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
)
