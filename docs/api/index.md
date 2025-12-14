# API Reference

This section provides API documentation for all Chemeleon2 modules.

## Module Index

### Core Modules

| Module | Description | Source |
|--------|-------------|--------|
| `src.vae_module` | Variational Autoencoder for crystal structure encoding | [`src/vae_module/`](https://github.com/hspark1212/chemeleon2/tree/main/src/vae_module) |
| `src.ldm_module` | Latent Diffusion Model for generation | [`src/ldm_module/`](https://github.com/hspark1212/chemeleon2/tree/main/src/ldm_module) |
| `src.rl_module` | Reinforcement Learning fine-tuning | [`src/rl_module/`](https://github.com/hspark1212/chemeleon2/tree/main/src/rl_module) |

### Data & Utilities

| Module | Description | Source |
|--------|-------------|--------|
| `src.data` | Data loading and processing | [`src/data/`](https://github.com/hspark1212/chemeleon2/tree/main/src/data) |
| `src.utils` | Metrics, featurization, and visualization | [`src/utils/`](https://github.com/hspark1212/chemeleon2/tree/main/src/utils) |

## Quick Reference

### Main Classes

| Class | Module | Key Methods | Description |
|-------|--------|-------------|-------------|
| `VAEModule` | `src.vae_module.vae_module` | `encode()`, `decode()`, `sample()`, `reconstruct()` | Lightning module for VAE |
| `LDMModule` | `src.ldm_module.ldm_module` | `calculate_loss()`, `sample()` | Lightning module for LDM |
| `RLModule` | `src.rl_module.rl_module` | `rollout()`, `compute_rewards()`, `calculate_loss()` | Lightning module for RL |
| `DataModule` | `src.data.datamodule` | `setup()`, `train_dataloader()` | Lightning DataModule |
| `CrystalBatch` | `src.data.schema` | `to_atoms()`, `to_structure()`, `collate()` | Batch container for crystals |
| `Metrics` | `src.utils.metrics` | `compute()`, `to_dataframe()`, `to_csv()` | Evaluation metrics |
| `ReinforceReward` | `src.rl_module.reward` | `forward()`, `normalize()` | Aggregates multiple reward components |
| `PredictorModule` | `src.vae_module.predictor_module` | `predict()` | Property predictor in latent space |

### Training Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `src/train_vae.py` | Train VAE model | `python src/train_vae.py experiment=mp_20/vae_dng` |
| `src/train_ldm.py` | Train LDM model | `python src/train_ldm.py experiment=mp_20/ldm_null` |
| `src/train_rl.py` | Train RL model | `python src/train_rl.py experiment=mp_20/rl_dng` |
| `src/train_predictor.py` | Train property predictor | `python src/train_predictor.py experiment=alex_mp_20_bandgap/predictor` |
| `src/sample.py` | Generate structures | `python src/sample.py --num_samples=1000` |
| `src/evaluate.py` | Evaluate generated structures | `python src/evaluate.py --structure_path=outputs/structures.json.gz` |

### Reward Components

| Component | Module | Description |
|-----------|--------|-------------|
| `CustomReward` | `src.rl_module.components` | User-defined reward function (override `compute()`) |
| `PredictorReward` | `src.rl_module.components` | Surrogate model predictions |
| `CreativityReward` | `src.rl_module.components` | Uniqueness + Novelty with AMD fallback |
| `EnergyReward` | `src.rl_module.components` | Energy above hull minimization (requires mace-torch) |
| `StructureDiversityReward` | `src.rl_module.components` | MMD-based structure diversity |
| `CompositionDiversityReward` | `src.rl_module.components` | MMD-based composition diversity |

## Usage Examples

### Load Pre-trained Models

```python
from src.vae_module import VAEModule
from src.ldm_module import LDMModule

# Load from checkpoint
vae = VAEModule.load_from_checkpoint(
    "path/to/vae.ckpt",
    weights_only=False
)

# Load LDM with VAE
ldm = LDMModule.load_from_checkpoint(
    "path/to/ldm.ckpt",
    vae_ckpt_path="path/to/vae.ckpt",
    weights_only=False
)
```

### Generate Structures (CLI)

```bash
# Generate 1000 structures using DDIM sampler
python src/sample.py \
    --num_samples=1000 \
    --batch_size=500 \
    --sampler=ddim \
    --sampling_steps=50 \
    --output_dir=outputs

# Generate for specific compositions (CSP task)
python src/sample.py \
    --num_samples=10 \
    --compositions="LiFePO4,Li2Co2O4,LiMn2O4,LiNiO2"
```

### Generate Structures (Programmatic)

```python
from src.ldm_module import LDMModule
from src.data.schema import create_empty_batch
import torch

# Load model
ldm = LDMModule.load_from_checkpoint(
    "path/to/ldm.ckpt",
    vae_ckpt_path="path/to/vae.ckpt",
    weights_only=False
)
ldm.eval()

# Create batch with desired number of atoms
num_atoms = torch.tensor([10, 12, 15])  # 3 structures
batch = create_empty_batch(num_atoms, device="cuda")

# Sample structures
batch_gen = ldm.sample(batch, sampling_steps=50)
structures = batch_gen.to_structure()  # Convert to pymatgen Structure
```

### Evaluate Structures

```python
from src.utils.metrics import Metrics
from monty.serialization import loadfn

# Load generated structures
gen_structures = loadfn("outputs/structures.json.gz")

# Create metrics object
metrics = Metrics(
    reference_dataset="mp-20",  # or "mp-all", "alex-mp-20"
    phase_diagram="mp-all",
    metastable_threshold=0.1
)

# Compute metrics
results = metrics.compute(gen_structures=gen_structures)

# Print results
print(f"Uniqueness: {results['avg_unique']:.2%}")
print(f"Novelty: {results['avg_novel']:.2%}")
print(f"Avg Energy Above Hull: {results['avg_e_above_hull']:.3f} eV/atom")

# Save to CSV
metrics.to_csv("results.csv")
```

### Evaluate Structures (CLI)

```bash
# Evaluate from JSON file
python src/evaluate.py \
    --structure_path=outputs/structures.json.gz \
    --reference_dataset=mp-20 \
    --output_file=results.csv

# Generate and evaluate in one command
python src/evaluate.py \
    --model_path=path/to/ldm.ckpt \
    --structure_path=outputs \
    --num_samples=1000
```

### Custom Reward Component

```python
from src.rl_module.components import RewardComponent
import torch

class MyCustomReward(RewardComponent):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(weight=weight, **kwargs)
        # Initialize any parameters

    def compute(self, gen_structures, **kwargs):
        """Compute rewards for generated structures.

        Args:
            gen_structures: list[Structure] - Generated structures
            **kwargs: Additional arguments (batch_gen, metrics_obj, device, etc.)

        Returns:
            torch.Tensor of shape (num_structures,)
        """
        rewards = []
        for structure in gen_structures:
            # Your reward logic here
            reward = self.calculate_reward(structure)
            rewards.append(reward)

        return torch.tensor(rewards)
```

### Featurize Structures

```python
from src.utils.featurizer import featurize

# Featurize structures using pre-trained VAE
features = featurize(
    structures=[structure1, structure2, structure3],
    model_path=None,  # Uses default VAE from HF Hub
    batch_size=2000,
    device="cuda"
)

# Access features
structure_features = features["structure_features"]  # (N, latent_dim)
composition_features = features["composition_features"]  # (N, embed_dim)
atom_features = features["atom_features"]  # list of (num_atoms, latent_dim)
```

## Configuration

All modules use [Hydra](https://hydra.cc/) for configuration management.

### CLI Configuration

```bash
# Override config via CLI
python src/train_vae.py experiment=mp_20/vae_dng trainer.max_epochs=100

# Multiple overrides
python src/train_vae.py \
    experiment=mp_20/vae_dng \
    data.batch_size=256 \
    trainer.max_epochs=100

# Show full config
python src/train_vae.py experiment=mp_20/vae_dng --cfg job

# Show available experiments
ls configs/experiment/
```

## Learn More

- [Architecture Overview](../architecture/index.md) - System design and components
- [Training Guide](../user-guide/training/index.md) - Step-by-step training tutorials
- [Custom Rewards](../user-guide/rewards/index.md) - Creating custom reward functions
- [Evaluation Guide](../user-guide/evaluation.md) - Metrics and benchmarking
