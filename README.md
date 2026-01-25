#AI STM - Environment Setup 

# Table of Contents
* [Introduction](#intro)
* [Installation](#install)
* [Environment Verification](#environmentverify)

# Introduction
This environment targets Ubuntu-based Linux systems and is GPU-ready once CUDA and PyTorch are installed. The current
setup is designed to run on CPU-only systems as well.

# Installation
This project uses **Conda** for environment management. The environment definition is provided in 'environment.yml'.
To create the Environment: 
```bash
conda env create -f environment.yml -n ai-stm
conda activate ai-stm
```
To update environment after modifications: 
```bash
conda env update -f environment.yml -n ai-stm --prune
```

# Environment Verification 
In order to verify that the environment is correctly configured for local or cloud execution, from the project directory run:
```bash
python verify/verify_cuda.py
```

