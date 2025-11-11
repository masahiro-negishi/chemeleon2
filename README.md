
# Chemeleon2

A reinforcement learning framework in latent diffusion models for crystal structure generation using group relative policy optimization.

<p align="center">
  <img src="assets/logo.png" alt="Chemeleon2 logo" width="260">
</p>

## Overview

Chemeleon2 implements a three-stage pipeline for crystal structure generation:

1. **VAE Module**: Encodes crystal structures into latent space representations
2. **LDM Module**: Generates structures in latent space using diffusion
3. **RL Module**: Fine-tunes the LDM with reward functions

<p align="center">
  <img src="assets/toc.png" alt="Chemeleon2 pipeline overview" width="640">
</p>

## Installation

```bash
# Clone the repository
git clone https://github.com/hspark1212/chemeleon2
cd chemeleon2

# Install dependencies with uv
uv sync
```

> **Tip:** `uv sync` installs dependencies based on the `uv.lock` file, ensuring reproducible environments. If you encounter issues with `uv.lock` (e.g., lock file conflicts or compatibility problems), you can use `uv pip install -e .` as an alternative to install the package in editable mode directly from `pyproject.toml`.

### (Optional) Installation with dependency

```bash
# (Optional) Install development dependencies (pytest, ruff, pyright, etc.)
uv sync --extra dev

# (Optional) Install metrics dependencies for evaluation (mace-torch, smact)
uv sync --extra metrics
```

### (Optional) Pytorch Installation with CUDA

After completing `uv sync`, install a PyTorch version compatible with your CUDA environment to prevent compatibility issues.
For version-specific installation commands, visit the [PyTorch official website](https://pytorch.org/get-started/previous-versions/).

<details>
<summary> Example </summary>
For PyTorch 2.7.0 with CUDA 12.8:
<pre><code>uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128</code></pre>

</details>

## Quick Start

For a simple walkthrough of sampling and evaluation, see [tutorial.ipynb](./tutorial.ipynb).

## Training

Chemeleon2 uses a three-stage training pipeline: VAE → LDM → RL.

For detailed instructions, see:

- [Training Guide](docs/TRAINING.md) - VAE, LDM, RL, and predictor training
- [Evaluation Guide](docs/EVALUATION.md) - Sampling and model evaluation/metrics

## Benchmarks

To benchmark de novo generation (DNG), 10,000 sampled structures are available in the `benchmarks/dng/` directory:

- **MP-20**: [`chemeleon2_rl_dng_mp_20.json.gz`](benchmarks/dng/chemeleon2_rl_dng_mp_20.json.gz) - 10,000 generated structures using RL-trained model on MP-20
- **Alex-MP-20**: [`chemeleon2_rl_dng_alex_mp_20.json.gz`](benchmarks/dng/chemeleon2_rl_dng_alex_mp_20.json.gz) - 10,000 generated structures using RL-trained model on Alex-MP-20
The compressed json files can be load them using `from monty.serialization import loadfn`.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed setup instructions, development workflow, and guidelines.
