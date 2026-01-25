# AI STM - Environment Setup 

## Table of Contents
## Table of Contents
* [Introduction](#introduction)
* [Environment Prerequisites](#environment-prerequisites)
* [Installation](#installation)
* [Environment Verification](#environment-verification)

## Introduction
This environment targets Ubuntu-based Linux systems and is intended to be GPU-enabled once a CUDA-capable PyTorch build is installed. The current
setup is designed to run on CPU-only systems as well. Once Ubuntu/GPU access is available, the setup procedure below will be executed as is and all version numbers and dependency resolutions will be finalized.

## Environment Prerequisites
- A Unix-like operating system (Ubuntu 20.04+ for GPU execution)
- Conda (Miniconda or equivalent)
- Python (managed via Conda, no system Python installation required)
This project assumes a working Conda installation. For setup instructions, see: [link](https://docs.conda.io/en/latest/miniconda.html)

## Installation
This project uses **Conda** for environment management. The environment definition is provided in 'environment.yml'. Clone this repository and follow the steps below to create the Conda environment.

To create the Environment: 
```bash
conda env create -f environment.yml -n ai-stm
conda activate ai-stm
```
To update environment after modifications: 
```bash
conda env update -f environment.yml -n ai-stm --prune
```

PyTorch is installed separately because the correct build depends on OS and (for Linux) the target CUDA version.

Local development (CPU-only)

For local, CPU-only environments (e.g. macOS, Windows, or Linux without CUDA):
```bash
conda install -y pytorch
```

## Ubuntu 

On Ubuntu with an NVIDIA GPU, install a CUDA-enabled PyTorch build appropriate for the machine’s CUDA driver/toolkit version. Exact version will be finalized once the target GPU machine configuration is confirmed.

## Environment Verification 

In order to verify that the environment is correctly configured for local or cloud execution, from the project directory run:
```bash
python verify/verify_cuda.py
```
On local CPU-only environment, CUDA available: False is expected. 

