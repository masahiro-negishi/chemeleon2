# Tutorial: Predictor-Based Reward

Learn to train a property predictor and use it as a reward signal for RL fine-tuning.

## Overview

This tutorial covers the complete workflow:
1. Prepare a dataset with property labels
2. Train a property predictor
3. Configure PredictorReward for RL
4. Run RL training with property optimization

**Use case:** Optimize generated structures for band gap targeting 3.0 eV.

## Prerequisites

- Dataset with property labels (e.g., band gap, energy above hull)

## Step 1: Prepare Your Dataset

### Option A: Use Existing Dataset

The MP-20 dataset includes band gap labels and is ready to use:

```bash
ls data/mp-20/
# train.csv  val.csv  test.csv
```

:::{note}
Before proceeding, make sure your dataset CSV files contain the target property columns you want to predict (e.g., `band_gap`, `e_above_hull`). You can verify this by checking the column names:

```python
import pandas as pd
df = pd.read_csv("data/mp-20/train.csv")
print(df.columns.tolist())
# Should include: ['material_id', 'cif', 'band_gap', 'e_above_hull', ...]
```
:::

### Option B: Create Custom Dataset

Query Materials Project for structures with band gap labels:

```python
import pandas as pd
from mp_api.client import MPRester

with MPRester(api_key="YOUR_API_KEY") as mpr:
    docs = mpr.materials.summary.search(
        num_sites=[1, 20],
        energy_above_hull=[0, 0.1],  # Stable structures
        fields=["material_id", "structure", "band_gap", "energy_above_hull"],
    )

data = []
for doc in docs:
    data.append({
        "material_id": doc.material_id,
        "band_gap": doc.band_gap,
        "e_above_hull": doc.energy_above_hull,  # Save as e_above_hull to match MP-20 format
        "cif": doc.structure.to(fmt="cif"),
    })

df = pd.DataFrame(data)
print(f"Collected {len(df)} structures")

# Compute normalization statistics (needed for config)
print(f"Band gap mean: {df['band_gap'].mean():.3f}")
print(f"Band gap std: {df['band_gap'].std():.3f}")
print(f"E above hull mean: {df['e_above_hull'].mean():.3f}")
print(f"E above hull std: {df['e_above_hull'].std():.3f}")

# Split into train/val/test
train_df = df.sample(frac=0.8, random_state=42)
remaining = df.drop(train_df.index)
val_df = remaining.sample(frac=0.5, random_state=42)
test_df = remaining.drop(val_df.index)

# Save
train_df.to_csv("data/my_bandgap/train.csv", index=False)
val_df.to_csv("data/my_bandgap/val.csv", index=False)
test_df.to_csv("data/my_bandgap/test.csv", index=False)
```

## Step 2: Train the Predictor

### Create Predictor Configuration

Reference file: [`configs/custom_reward/predictor_band_gap.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/custom_reward/predictor_band_gap.yaml)

```yaml
# @package _global_
# Predictor training for band gap on MP-20

data:
  _target_: src.data.datamodule.DataModule
  data_dir: ${paths.data_dir}/mp-20
  batch_size: 256
  dataset_type: "mp-20"
  target_condition: band_gap

predictor_module:
  vae:
    checkpoint_path: ${hub:mp_20_vae}

  target_conditions:
    band_gap:
      mean: 0.792   # Dataset statistics
      std: 1.418

logger:
  wandb:
    name: "predictor_band_gap"
```

```{tip}
The `target_condition` name (e.g., `band_gap`) must match exactly with:
- The column name in your CSV data files
- The key in `target_conditions`
- The `target_name` in PredictorReward config or your custom reward class
```

:::{note}
The `${hub:mp_20_vae}` syntax automatically downloads pre-trained models from HuggingFace. See [Automatic Download from HuggingFace](../training/index.md#automatic-download-from-huggingface) for details.
:::

### Run Predictor Training

```bash
# Using the MP-20 dataset
python src/train_predictor.py custom_reward=predictor_band_gap

# Or for custom dataset
python src/train_predictor.py experiment=my_bandgap/predictor
```

### Verify Predictor Quality

:::{tip}
**Example training run**: See [this W&B run](https://wandb.ai/hspark1212/chemeleon2_project/runs/l3q53ozk) for predictor training example with band gap on MP-20.
:::


## Step 3: Configure RL with Predictor Reward

### Create Custom Reward Class

Create a custom reward class that implements predictor-based reward:

```python
# Reference: custom_reward/predictor_bandgap.py
"""Band gap predictor-based reward for RL training."""

import torch

from src.data.schema import CrystalBatch
from src.rl_module.components import RewardComponent
from src.vae_module.predictor_module import PredictorModule


class BandGapPredictorReward(RewardComponent):
    """Reward based on predicted band gap value.

    This reward uses a trained predictor to estimate band gap values and
    optimizes structures toward a target band gap value (e.g., 3.0 eV for
    wide band gap semiconductors).
    """

    required_metrics = []

    def __init__(
        self,
        predictor: PredictorModule,
        target_value: float = 3.0,
        clip_min: float = 0.0,
        clip_max: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.predictor = predictor
        self.target_name = "band_gap"
        self.target_value = target_value
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.predictor.eval()

    def compute(self, batch_gen: CrystalBatch, **kwargs) -> torch.Tensor:
        """Compute reward based on predicted band gap values."""
        device = self.predictor.device
        batch_gen = batch_gen.to(device)

        # Get predictions from the predictor
        pred = self.predictor.predict(batch_gen)
        pred_val = pred[self.target_name].clamp(min=self.clip_min, max=self.clip_max)

        # Compute reward based on target value
        if self.target_value is not None:
            # Negative squared error: closer to target = higher reward
            reward = -((pred_val - self.target_value) ** 2)
        else:
            # No target: maximize predicted value
            reward = pred_val

        return reward
```

### Create RL Configuration

Reference file: [`configs/custom_reward/rl_bandgap.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/custom_reward/rl_bandgap.yaml)

```yaml
# @package _global_
# RL Custom Reward Experiment Configuration
#
# Example: Band gap predictor-based reward using BandGapPredictorReward
# See user guide: docs/user-guide/rewards/predictor-reward.md

data:
  data_dir: ${paths.data_dir}/mp-20
  batch_size: 5

trainer:
  max_steps: 1000
  strategy: ddp_find_unused_parameters_true  # Required when using predictor-based rewards

rl_module:
  ldm_ckpt_path: ${hub:mp_20_ldm_base}
  vae_ckpt_path: ${hub:mp_20_vae}

  rl_configs:
    num_group_samples: 64
    group_reward_norm: true

  reward_fn:
    normalize_fn: std
    eps: 1e-4
    reference_dataset: mp-20
    components:
      - _target_: custom_reward.predictor_bandgap.BandGapPredictorReward
        weight: 1.0
        normalize_fn: norm
        predictor:
          _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
          checkpoint_path: "ckpts/mp_20/predictor/predictor_band_gap.ckpt" # Change your checkpoint path
          map_location: "cpu"
        target_value: 3.0  # Target: wide band gap (3.0 eV)
        clip_min: 0.0      # Band gap cannot be negative
      - _target_: src.rl_module.components.CompositionDiversityReward
        weight: 0.5
        normalize_fn: norm

logger:
  wandb:
    name: rl_bandgap
```

```{tip}
**Key Points:**
- Update `checkpoint_path` to point to your trained predictor model
- Use `strategy: ddp_find_unused_parameters_true` to handle predictor parameters in DDP mode
- Predictor returns **denormalized** values automatically - no need to manually scale
- Add `CompositionDiversityReward` to encourage diverse chemical exploration
- The `target_value` parameter controls optimization behavior (3.0 eV for wide bandgap semiconductors)
```

## Step 4: Run RL Training

```bash
# Use the default checkpoint path in config
python src/train_rl.py custom_reward=rl_bandgap

# Using custom reward with predictor checkpoint from CLI
python src/train_rl.py custom_reward=rl_bandgap \
  rl_module.reward_fn.components.0.predictor.checkpoint_path="your/path/to/predictor.ckpt"
```

Training script: [`src/train_rl.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_rl.py)

:::{tip}
**Example training run**: See [this W&B run](https://wandb.ai/hspark1212/chemeleon2_project/runs/29wy1301) for RL training example with band gap predictor on MP-20.
:::

## Advanced Configuration

### Maximize vs. Target Value

**Maximize band gap** (no upper bound):
```yaml
- _target_: custom_reward.predictor_bandgap.BandGapPredictorReward
  predictor:
    _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
    checkpoint_path: "ckpts/mp_20/predictor/predictor_band_gap.ckpt"
    map_location: "cpu"
  target_value: null  # No target = maximize
  clip_min: 0.0
```

**Target specific value** (e.g., 2.5 eV):
```yaml
- _target_: custom_reward.predictor_bandgap.BandGapPredictorReward
  predictor:
    _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
    checkpoint_path: "ckpts/mp_20/predictor/predictor_band_gap.ckpt"
    map_location: "cpu"
  target_value: 2.5  # Penalize deviation from 2.5 eV
```

### Multi-Objective with Predictor

Combine band gap optimization with stability and diversity:

```yaml
components:
  - _target_: custom_reward.predictor_bandgap.BandGapPredictorReward
    weight: 1.0
    normalize_fn: norm
    predictor:
      _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
      checkpoint_path: "ckpts/mp_20/predictor/predictor_band_gap.ckpt"
      map_location: "cpu"
    target_value: 3.0
    clip_min: 0.0

  - _target_: src.rl_module.components.EnergyReward
    weight: 0.5
    normalize_fn: norm

  - _target_: src.rl_module.components.CompositionDiversityReward
    weight: 0.5
    normalize_fn: norm
```

### Multiple Properties

Train predictor for multiple targets:

```yaml
# In predictor config
predictor_module:
  target_conditions:
    band_gap:
      mean: 0.792
      std: 1.418
    e_above_hull:
      mean: 0.035   # Computed from your dataset
      std: 0.028
```

You can create multiple custom reward classes for different properties and use them together:

```yaml
components:
  - _target_: custom_reward.predictor_bandgap.BandGapPredictorReward
    weight: 1.0
    predictor:
      _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
      checkpoint_path: "ckpts/multi_property_predictor.ckpt"
      map_location: "cpu"
    target_value: 3.0
    clip_min: 0.0

  - _target_: custom_reward.predictor_energy_above_hull.EnergyAboveHullReward
    weight: 0.5
    predictor:
      _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
      checkpoint_path: "ckpts/multi_property_predictor.ckpt"
      map_location: "cpu"
    target_value: 0.0  # Target stable structures (0 eV/atom above hull)
    clip_min: 0.0
```

## Reference Files

Working examples:
- Custom reward implementation: [`custom_reward/predictor_bandgap.py`](https://github.com/hspark1212/chemeleon2/blob/main/custom_reward/predictor_bandgap.py)
- Predictor config: [`configs/custom_reward/predictor_band_gap.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/custom_reward/predictor_band_gap.yaml)
- Custom reward config: [`configs/custom_reward/rl_bandgap.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/custom_reward/rl_bandgap.yaml)

## Summary

1. **Prepared dataset** with property labels
2. **Trained predictor** in VAE latent space
3. **Configured PredictorReward** with target value
4. **Combined with other rewards** for robust optimization

The predictor-based approach enables efficient property optimization without expensive DFT calculations during training.

## Next Steps

- [DNG Reward](dng-reward.md) - Multi-objective optimization
- [Atomic Density](atomic-density.md) - Simple custom reward
- [Training Overview](../training/index.md) - General training guide
