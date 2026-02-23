# AI STM - Environment Setup 

## Table of Contents
* [Introduction](#introduction)
* [Environment Prerequisites](#environment-prerequisites)
* [Easy Installation](#easy-installation)
* [(Optional) Multi-Step Installation](#installation)
* [(Optional) Environment Verification](#environment-verification)
* [(Optional) Assessment Pipeline Validation](#assessment-pipeline-validation)

## Introduction
The setup supports CPU-only execution on local machines and is designed to transition cleanly to GPU-enabled execution on Ubuntu systems with NVIDIA GPUs once CUDA-capable hardware is available. GPU-specific dependency versions (e.g., CUDA and PyTorch builds) will be finalized after confirming the target hardware configuration. Notes: Data and Models are not included in this repository.

## Environment Prerequisites
- A Unix-like operating system (Ubuntu 20.04+ for GPU execution)
- Conda (Miniconda or equivalent)
- Python (managed via Conda, no system Python installation required)
This project assumes a working Conda installation. For setup instructions, see: [link](https://docs.conda.io/en/latest/miniconda.html)

## Easy Installation
This project uses **Conda** for environment management. The environment definition is provided in 'environment.base.yml'. Clone this repository and paste the commands below at the file root, to create the Conda environment. This is the recommended installation path for most users.

```bash
conda env create -f environment.base.yml
conda activate ai-stm
python -m pip install -e .
ai-stm-check
```

## Installation
The below installation steps are provided for users who prefer a more explicit and manual setup process. The following steps are equivalent to the Easy Installation. 

To create the Environment: 
```bash
conda env create -f environment.base.yml -n ai-stm
conda activate ai-stm
```
To update environment after modifications: 
```bash
conda env update -f environment.base.yml -n ai-stm --prune
```

PyTorch is installed separately because the correct build depends on OS and (for Linux) the target CUDA version.

## Local development (CPU-only)

PyTorch is installed via Conda on CPU-only systems to avoid runtime conflicts, Important: Do not install GPU-enabled PyTorch on CPU-only systems, as this may cause OpenMP or runtime conflicts.

For local, CPU-only environments:
```bash
conda install -y pytorch
```

## Ubuntu + NVIDIA GPU (CUDA)

For systems with an NVIDIA GPU and CUDA support (Ubuntu-based servers or cloud instances), use the GPU-enabled environment. The exact CUDA and PyTorch versions will be finalized once the target hardware configuration is confirmed.

To create the GPU-enabled environment, run:
```bash 
conda env create -f environment.gpu.yml -n ai-stm
conda activate ai-stm
```

## Environment Verification 

In order to verify that the environment is correctly configured for local or cloud execution, from the project directory run:
```bash
python verify/verify_cuda.py
```
On local CPU-only environment, CUDA available: False is expected. 

## Assessment Pipeline Validation

This repository currently includes a minimal **assessment pipeline validation** to verify that the machine learning stack is correctly configured and executable. This can be run on local machine and reports GPU/CUDA availability, running on CPU when unavailable. 

The validation script:
- Uses a minimal CNN implemented in **PyTorch**
- Confirms CPU/GPU device selection and CUDA visibility (when available)
- Runs a forward pass and backward pass to ensure end-to-end training capability

This validation is **infrastructure-focused** and does not represent the finalized assessment model.  
The original DeepSPM reference implementation uses TensorFlow; here, PyTorch is used for modern CUDA/cloud compatibility and rapid iteration. The assessment architecture will be finalized once the dataset and training strategy are selected.

Run the validation
```bash
python assessment/scripts/validate_cnn_pipeline.py
```