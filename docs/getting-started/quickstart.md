# Quick Start

This guide walks you through generating crystal structures using pre-trained models.

## Download Pre-trained Models

Chemeleon2 provides pre-trained checkpoints via HuggingFace Hub:

| Model | Dataset | Description |
|-------|---------|-------------|
| `mp_20_vae` | MP-20 | VAE for Materials Project (20 atoms) |
| `mp_20_ldm` | MP-20 | LDM for de novo generation |
| `alex_mp_20_vae` | Alex-MP-20 | VAE for Alexandria dataset |
| `alex_mp_20_ldm` | Alex-MP-20 | LDM for de novo generation |

## Generate Crystal Structures

### Using Python API

```python
from src.sample import sample_structures

# Sample 100 structures using pre-trained model
structures = sample_structures(
    vae_ckpt="hspark1212/chemeleon2_mp_20_vae",
    ldm_ckpt="hspark1212/chemeleon2_mp_20_ldm",
    num_samples=100,
)

# Each structure is a pymatgen Structure object
for struct in structures[:5]:
    print(struct.composition)
```

### Using Command Line

```bash
python src/sample.py \
    vae_ckpt=hspark1212/chemeleon2_mp_20_vae \
    ldm_ckpt=hspark1212/chemeleon2_mp_20_ldm \
    num_samples=100 \
    output_path=generated_structures.json.gz
```

## Evaluate Generated Structures

```python
from src.utils.metrics import Metrics

# Initialize metrics calculator
metrics = Metrics(
    reference_structures=reference_data,  # Training set for novelty check
)

# Compute metrics
results = metrics.compute(generated_structures)

print(f"Uniqueness: {results['unique']:.2%}")
print(f"Novelty: {results['novel']:.2%}")
print(f"Energy above hull: {results['e_above_hull']:.3f} eV/atom")
```

## Interactive Tutorial

For a more detailed walkthrough, see the [tutorial.ipynb](https://github.com/hspark1212/chemeleon2/blob/main/tutorial.ipynb) notebook.

## Next Steps

- [Training Guide](../user-guide/training.md) - Train custom models
- [Evaluation Guide](../user-guide/evaluation.md) - Detailed metrics explanation
- [Custom Rewards](../user-guide/custom-rewards.md) - Define custom RL objectives
