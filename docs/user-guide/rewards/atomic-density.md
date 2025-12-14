# Tutorial: Atomic Density Reward

Learn to create a custom reward that maximizes atomic density in generated crystals.

## Objective

Create a reward function that encourages denser crystal structures:

$$\text{density} = \frac{M_{\text{total}}}{V_{\text{cell}}} \quad [\text{g/cm}^3]$$

Higher density = more mass packed per unit volume.


## Step 1: Understand the CustomReward Class

The `CustomReward` class in [`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py) is a placeholder for user-defined logic:

```python
class CustomReward(RewardComponent):
    """Wrapper for user-defined custom reward functions."""

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        """Placeholder for custom reward function."""
        return torch.zeros(len(gen_structures))
```

The `compute()` method receives:
- `gen_structures`: List of pymatgen `Structure` objects
- Additional kwargs like `batch_gen`, `device`, `metrics_obj`

## Step 2: Implement Atomic Density Reward

Edit [`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py) and modify the `CustomReward` class:

```python
class CustomReward(RewardComponent):
    """Atomic density reward - maximize atoms per unit volume."""

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        """
        Compute atomic density for each structure.

        Returns higher rewards for denser structures.
        """
        rewards = []
        for structure in gen_structures:
            density = structure.density  # atomic mass / volume [g/cm³]
            rewards.append(density)
        return torch.as_tensor(rewards)
```

## Step 3: Create Configuration File

See [`configs/custom_reward/atomic_density.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/custom_reward/atomic_density.yaml):

```yaml
# @package _global_
# RL Custom Reward Experiment Configuration

data:
  data_dir: ${paths.data_dir}/mp-20
  batch_size: 5

trainer:
  max_steps: 200

rl_module:
  ldm_ckpt_path: ${hub:mp_20_ldm_base}
  vae_ckpt_path: ${hub:mp_20_vae}

  rl_configs:
    num_group_samples: 64
    group_reward_norm: true

  reward_fn:
    normalize_fn: std
    components:
      - _target_: custom_reward.atomic_density.AtomicDensityReward

logger:
  wandb:
    name: rl_custom_reward
```

:::{note}
The `${hub:...}` syntax automatically downloads pre-trained models from HuggingFace. See [Automatic Download from HuggingFace](../training/index.md#automatic-download-from-huggingface) for details.
:::

## Step 4: Run Training

```bash
python src/train_rl.py custom_reward=atomic_density
```

Training script: [`src/train_rl.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_rl.py)

## Step 5: Monitor Training

:::{tip}
**Example training run**: See [this W&B run](https://wandb.ai/hspark1212/chemeleon2_project/runs/71u0nz7w) for atomic density reward training examples.
:::

In WandB, watch these metrics:

| Metric | Description |
|--------|-------------|
| `train/reward` | Mean reward from reward function (should increase) |
| `val/reward` | Validation reward |
| `train/advantages` | Normalized rewards used for policy gradient |
| `train/kl_div` | KL divergence from reference policy |
| `train/entropy` | Policy entropy |
| `train/loss` | Total policy loss |

As training progresses, the model should generate increasingly dense structures.

## Step 6: Evaluate Results

### Generate Samples

```bash
python src/sample.py \
    --ldm_ckpt_path=logs/train_rl/runs/<your-run>/checkpoints/last.ckpt \
    --num_samples=10 \
    --output_dir=outputs/rl_samples
```

### Analyze Density

```python
from monty.serialization import loadfn
import numpy as np

structures = loadfn("outputs/rl_samples/generated_structures.json.gz")
densities = [s.density for s in structures]

print(f"Mean density: {np.mean(densities):.3f} g/cm³")
print(f"Max density:  {np.max(densities):.3f} g/cm³")
```

:::{tip}
If Mean density is higher than 7 g/cm³, your first RL learning with your custom reward is successful!
:::

## Extensions

### Target Density

Instead of maximizing density, optimize toward a specific target. Create `custom_reward/target_density.py`:

```python
"""Target density reward for RL training."""

import torch
from pymatgen.core import Structure

from src.rl_module.components import RewardComponent


class TargetDensityReward(RewardComponent):
    """Reward based on distance from target density."""

    def __init__(self, target_density: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.target_density = target_density

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        rewards = []
        for structure in gen_structures:
            density = len(structure) / structure.lattice.volume
            # Negative distance from target (higher = closer to target)
            reward = -abs(density - self.target_density)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)
```

Create a config file (`configs/custom_reward/rl_target_density.yaml`):
```yaml
# @package _global_
rl_module:
  reward_fn:
    components:
      - _target_: custom_reward.target_density.TargetDensityReward
        target_density: 5.0  # atoms/Å³
```

### Combined with built-in Reward Components

Ensure dense structures are also stable by adding `EnergyReward` and `StructureDiversityReward`:

```yaml
# @package _global_
rl_module:
  reward_fn:
    components:
      - _target_: custom_reward.atomic_density.AtomicDensityReward
        weight: 1.0
        normalize_fn: norm
      - _target_: src.rl_module.components.EnergyReward
        weight: 0.5
        normalize_fn: norm
      - _target_: src.rl_module.components.StructureDiversityReward
        weight: 0.5
        normalize_fn: norm
```

This encourages the model to generate structures that are dense, low-energy, and diverse.


## Summary

1. Create your reward class in `custom_reward/` folder
2. Create config in `configs/custom_reward/` referencing your reward
3. Run training: `python src/train_rl.py custom_reward=your_config`
4. Combine with other components for multi-objective optimization

## Next Steps

- [DNG Reward](dng-reward.md) - Multi-objective optimization from the paper
- [Predictor Reward](predictor-reward.md) - Use ML models as reward
