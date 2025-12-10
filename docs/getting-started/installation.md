# Installation

## Prerequisites

- Python 3.11+
- Git
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Basic Installation

```bash
# Clone the repository
git clone https://github.com/hspark1212/chemeleon2
cd chemeleon2

# Install dependencies with uv
uv sync
```

:::{tip}
`uv sync` installs dependencies based on the `uv.lock` file, ensuring reproducible environments. If you encounter issues with `uv.lock` (e.g., lock file conflicts or compatibility problems), you can use `uv pip install -e .` as an alternative to install the package in editable mode directly from `pyproject.toml`.
:::

## Optional Dependencies

### Development Tools

```bash
# Install development dependencies (pytest, ruff, pyright, etc.)
uv sync --extra dev
```

### Metrics Dependencies

```bash
# Install metrics dependencies for evaluation (mace-torch, smact)
uv sync --extra metrics
```

## CUDA Support

After completing `uv sync`, install a PyTorch version compatible with your CUDA environment to prevent compatibility issues.

For version-specific installation commands, visit the [PyTorch official website](https://pytorch.org/get-started/previous-versions/).

```bash
# Example command for PyTorch 2.7.0 with CUDA 12.8
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

## Verify Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Run tests
pytest -v
```

## Next Steps

- [Quick Start](quickstart.md) - Run your first generation
- [Training Guide](../user-guide/training.md) - Train your own models
