"""
Board Games AI Training Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# 读取依赖
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("===")
        ]

dev_requirements_file = Path(__file__).parent / "requirements-dev.txt"
dev_requirements = []
if dev_requirements_file.exists():
    with open(dev_requirements_file) as f:
        dev_requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("===")
        ]

setup(
    name="board-games-ai",
    version="0.1.0-dev",
    author="Your Name",
    author_email="your.email@example.com",
    description="A general-purpose reinforcement learning framework for board games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Board_Games",
    packages=find_packages(exclude=["tests*", "docs*", "scripts*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "board-games-train=scripts.train:main",
            "board-games-eval=scripts.evaluate:main",
            "board-games-play=scripts.play_human:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml"],
    },
    zip_safe=False,
)
