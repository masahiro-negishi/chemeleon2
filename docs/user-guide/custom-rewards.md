# Custom Reward RL Training Guide

This guide explains how to train crystal structure generation models with custom reward functions using reinforcement learning.

## Overview

The RL module fine-tunes a pre-trained Latent Diffusion Model (LDM) to generate crystal structures that maximize user-defined reward functions. There are two main scenarios:

1. **Simple Custom Reward**: Define a reward function based on generated structures
2. **Predictor-Based Reward**: Train a property predictor first, then use it as a reward signal

## Quick Start: Simple Custom Reward

### Step 1: Create Your Custom Reward Component

Edit `src/rl_module/components.py` and modify the `CustomReward` class:

```python
class CustomReward(RewardComponent):
    """Wrapper for user-defined custom reward functions."""

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        """
        Compute rewards for generated crystal structures.

        Args:
            gen_structures: List of pymatgen Structure objects

        Returns:
            torch.Tensor of shape (num_structures,) with reward values
        """
        rewards = []
        for structure in gen_structures:
            # Example: Reward structures with specific number of atoms
            reward = -abs(len(structure) - 20)  # Target 20 atoms
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)
```

**Available inputs to `compute()`:**

- `gen_structures`: List of `pymatgen.core.Structure` objects
- `batch_gen`: `CrystalBatch` object with batched tensor data
- `metrics_obj`: `Metrics` object (if `required_metrics` is set)
- `device`: Current torch device

### Step 2: Configure the Experiment

Create a config file (e.g., `configs/experiment/my_custom_rl.yaml`). See `configs/experiment/rl_custom_reward.yaml` for a reference:

```yaml
# @package _global_
# Custom RL experiment
# Only override what's different from train_rl.yaml defaults
# (In this case, nothing needs to be overridden since we use mp-20 dataset)

data:
  batch_size: 5

rl_module:
  ldm_ckpt_path: ${hub:mp_20_ldm}
  vae_ckpt_path: ${hub:mp_20_vae}

  rl_configs:
    num_group_samples: 64
    group_reward_norm: true

  reward_fn:
    _target_: src.rl_module.reward.ReinforceReward
    normalize_fn: std
    eps: 1e-4
    reference_dataset: mp-20
    components:
      - _target_: src.rl_module.components.CustomReward
        weight: 1.0
        normalize_fn: norm

logger:
  wandb:
    name: my_custom_rl
```

### Step 3: Run Training

```bash
python src/train_rl.py experiment=my_custom_rl
```

## Advanced: Predictor-Based Reward

Use this approach when you want to optimize for a specific material property (e.g., band gap, formation energy). See `configs/experiment/alex_mp_20_bandgap/` for a complete working example.

### Step 1: Prepare Your Dataset

Create a dataset with property labels. See `data/mp-120/data_preparation.ipynb` for an example:

```python
import pandas as pd
from mp_api.client import MPRester

with MPRester(api_key) as mpr:
    docs = mpr.materials.summary.search(
        num_sites=[0, 120],
        energy_above_hull=[0, 0.1],
        fields=["material_id", "structure", "band_gap"],
    )

data = []
for doc in docs:
    data.append({
        "material_id": doc.material_id,
        "band_gap": doc.band_gap,
        "cif": doc.structure.to(fmt="cif"),
    })

df = pd.DataFrame(data)
# Split into train/val/test and save as CSV files
```

Place your CSV files in `data/your_dataset/` with columns: `material_id`, `cif`, and your target property.

### Step 2: Train the Predictor

Create a predictor config (e.g., `configs/experiment/my_dataset/predictor_bandgap.yaml`):

```yaml
# @package _global_
# Predictor training config
# Inherits defaults from train_predictor.yaml
# Define custom data config inline for specialized datasets

data:
  _target_: src.data.datamodule.DataModule
  data_dir: ${paths.data_dir}/my_dataset
  batch_size: 256
  dataset_type: "my_dataset"
  target_condition: band_gap
  num_workers: 16

predictor_module:
  vae:
    checkpoint_path: ${hub:mp_20_vae}

  target_conditions:
    band_gap:
      mean: 1.5   # Compute from your dataset
      std: 1.2    # Compute from your dataset

logger:
  wandb:
    name: "predictor_bandgap"
```

Train the predictor:

```bash
python src/train_predictor.py experiment=my_dataset/predictor_bandgap
```

### Step 3: Configure RL with Predictor Reward

Create RL config (e.g., `configs/experiment/my_dataset/rl_bandgap.yaml`):

```yaml
# @package _global_
# RL training with predictor reward
# Inherits defaults from train_rl.yaml
# Define custom data config inline for specialized datasets

data:
  _target_: src.data.datamodule.DataModule
  data_dir: ${paths.data_dir}/my_dataset
  batch_size: 5
  dataset_type: "my_dataset"
  target_condition: band_gap
  num_workers: 16

trainer:
  max_steps: 1000

rl_module:
  ldm_ckpt_path: ${hub:mp_20_ldm}
  vae_ckpt_path: ${hub:mp_20_vae}

  rl_configs:
    clip_ratio: 0.0001
    kl_weight: 1.0
    num_group_samples: 64
    group_reward_norm: true

  reward_fn:
    _target_: src.rl_module.reward.ReinforceReward
    normalize_fn: std
    eps: 1e-4
    reference_dataset: mp-20
    components:
      - _target_: src.rl_module.components.PredictorReward
        weight: 1.0
        normalize_fn: norm
        predictor:
          _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
          checkpoint_path: "ckpts/my_dataset/predictor/predictor_bandgap.ckpt"
          map_location: "cpu"
        target_name: band_gap
        target_value: 3.0    # Target band gap value
        clip_min: 0.0        # Optional: clip predictions

logger:
  wandb:
    name: rl_bandgap
```

Run RL training:

```bash
python src/train_rl.py experiment=my_dataset/rl_bandgap
```

## Built-in Reward Components

See `src/rl_module/components.py` for implementation details. You can combine multiple reward components:

| Component | Description | Required Metrics |
|-----------|-------------|------------------|
| `CustomReward` | User-defined reward function | None |
| `PredictorReward` | Property prediction from trained predictor | None |
| `CreativityReward` | Rewards unique and novel structures | `unique`, `novel` |
| `EnergyReward` | Penalizes high energy above hull | `e_above_hull` |
| `StructureDiversityReward` | Rewards diverse crystal structures | `structure_diversity` |
| `CompositionDiversityReward` | Rewards diverse chemical compositions | `composition_diversity` |

### Example: Multi-Objective Reward

```yaml
reward_fn:
  _target_: src.rl_module.reward.ReinforceReward
  normalize_fn: std
  reference_dataset: mp-20
  components:
    - _target_: src.rl_module.components.PredictorReward
      weight: 1.0
      normalize_fn: norm
      predictor:
        _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
        checkpoint_path: "ckpts/predictor.ckpt"
        map_location: "cpu"
      target_name: band_gap
      target_value: 2.5
    - _target_: src.rl_module.components.CompositionDiversityReward
      weight: 0.5
      normalize_fn: norm
    - _target_: src.rl_module.components.EnergyReward
      weight: 0.3
      normalize_fn: norm
```

## Normalization Options

Each component supports normalization via `normalize_fn` (see `src/rl_module/reward.py`):

- `norm`: Min-max normalization to [0, 1]
- `std`: Standardization (zero mean, unit variance)
- `subtract_mean`: Subtract mean only
- `clip`: Clip to [-1, 1]
- `null`: No normalization

## Tips

- Start with `num_group_samples: 64` and adjust based on GPU memory
- Use `group_reward_norm: true` for more stable training
- Monitor `val/reward` for early stopping
- The `weight` parameter controls relative importance of each component
- Use `clip_min`/`clip_max` in `PredictorReward` to bound predictions
