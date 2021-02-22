#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.md") as history_file:
    history = history_file.read()

# Common libraries
requirements = [
    "appdirs==1.4.4",
    "boto3==1.14.50",
    "Click>=6.0",
    "deprecation>=2.0.6",
    "dill==0.3.2",
    "docker==4.2.0",
    "GitPython==3.1.7",
    "google-api-python-client==1.12.8",
    "google-auth==1.23.0",
    "google-cloud==0.34.0",
    "google-cloud-container==2.3.0",
    "grpcio==1.34.0",
    "kubernetes==12.0.1",
    "lmdb==1.0.0",
    "matplotlib==3.2.1",
    "numpy==1.19.2",
    "oauth2client==4.1.3",
    "sklearn==0.0",
    "supermutes==0.2.5",
    "tabulate>=0.8.5",
    "tensorpack==0.10.1",
]

# Libraries used by torch
torch_reqs = [
    "sacrebleu==1.4.14",
    "torch==1.7.0",
    "torchvision==0.8.1",
]

tensorflow_reqs = [
    "tensorflow==1.13.2",
]

setup_requirements = [
    "pytest-runner",
]

lint_requirements = [
    "black==20.8b1",
    "isort==5.6.4",
]

test_requirements = (
    [
        "codecov==2.1.9",
        "coverage==5.3",
        "freezegun==1.0.0",
        "pre-commit",
        "pytest>=3",
        "pytest-cov==2.10.1",
        "pytest-mock==3.3.1",
        "wcwidth==0.2.5",
    ]
    + lint_requirements
    + torch_reqs
    + tensorflow_reqs
)

dev_requirements = torch_reqs + tensorflow_reqs + lint_requirements + test_requirements
extras = {
    "test": test_requirements,
    "lint": lint_requirements,
    "torch": torch_reqs,
    "tensorflow": tensorflow_reqs,
    "dev": dev_requirements,
}

setup(
    author="Ralf Grubenmann",
    author_email="ralf.grubenmann@epfl.ch",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="A public and reproducible collection of reference implementations and benchmark suite for distributed machine learning systems.",
    entry_points={
        "console_scripts": [
            "mlbench=mlbench_core.cli:cli_group",
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="mlbench",
    name="mlbench_core",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require=extras,
    url="https://github.com/mlbench/mlbench_core",
    version="3.0.0-dev13",
    zip_safe=False,
)
