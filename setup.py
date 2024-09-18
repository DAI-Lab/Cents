#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages
from setuptools import setup

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md", encoding="utf-8") as history_file:
    history = history_file.read()

install_requires = [
    "numpy>=1.23.5",
    "openai>=1.35.13",
    "pandas>=1.5.3",
    "matplotlib>=3.7.5",
    "scikit-learn>=1.1.3",
    "tiktoken>=0.7.0",
    "transformers>=4.44.0",
    "torch>=1.9.0",
    "accelerate>=0.32.1",
    "torchvision>=0.18.1",
    "tensorboard>=2.5.0",
    "tensorboardX>=2.6.2.2",
    "pyyaml>=6.0.1",
    "pre-commit>=3.5.0",
    "black>=24.4.2",
    "isort>=5.13.2",
    "dtaidistance>=2.3.12",
    "seaborn>=0.13.2",
    "einops>=0.8.0",
    "sentencepiece>=0.2.0",
    "omegaconf>=2.3.0",
]

setup_requires = [
    "pytest-runner>=2.11.1",
]

tests_require = [
    "pytest>=3.4.2",
    "pytest-cov>=2.6.0",
]

development_requires = [
    # general
    "bumpversion>=0.5.3",
    "pip>=9.0.1",
    "watchdog>=0.8.3",
    # docs
    # "m2r2>=0.2.0",
    # "Sphinx>=4.0.2,<6.0.0",
    # "sphinx_rtd_theme>=0.2.4,<0.5",
    # "autodocsumm>=0.1.10",
    # style check
    "flake8>=3.7.7",
    # "isort>=4.3.4",
    # fix style issues
    # "autoflake>=1.2",
    "autopep8>=1.4.3",
    # distribute on PyPI
    "twine>=1.10.0",
    "wheel>=0.30.0",
    # Advanced testing
    "coverage>=4.5.1",
    "tox>=2.9.1",
]

setup(
    author="Michael Fuest",
    author_email="fuest@mit.edu",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="A package for generative modeling and evaluation of synthetic household-level electricity load timeseries.",
    extras_require={
        "test": tests_require,
        "dev": development_requires + tests_require,
    },
    install_package_data=True,
    install_requires=install_requires,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="endata EnData EnData",
    name="EnData",
    packages=find_packages(include=["endata", "endata.*"]),
    python_requires=">=3.8",
    setup_requires=setup_requires,
    test_suite="tests",
    tests_require=tests_require,
    url="https://github.com/michael-fuest/EnData",
    version="0.1.0.dev0",
    zip_safe=False,
)
