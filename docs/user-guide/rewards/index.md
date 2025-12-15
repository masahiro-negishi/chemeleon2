# Custom Rewards Overview

This guide explains how to configure and customize reward functions for RL training in Chemeleon2.

## Why Verifiable Rewards?

Generative models for crystal structure generation face a fundamental **objective misalignment**: likelihood-based sampling inherently favors high-density regions of known compounds, while scientific discovery requires targeted exploration of underexplored regions where novel materials reside.

Reward functions enable the model to optimize for **verifiable scientific objectives** beyond likelihood maximization:

```{mermaid}
flowchart LR
    A[Generated Structure] --> B[Reward Components]
    B --> C[Total Reward]
    C --> D[Policy Update]
    D -.-> E[LDM]
    E -.-> A
```
For implementation details and the GRPO algorithm, see the [RL Module architecture guide](../../architecture/rl-module.md).

## Quick Start

```bash
# Run DNG reward training (multi-objective)
python src/train_rl.py experiment=mp_20/rl_dng

# Or with custom hyperparameters
python src/train_rl.py experiment=mp_20/rl_dng \
    rl_module.rl_configs.num_group_samples=128
```

## Quick Decision Guide

Choose the tutorial based on your use case:

| Use Case | Tutorial | Description |
|----------|----------|-------------|
| Simple custom logic | [Atomic Density](atomic-density.md) | Modify `CustomReward` class |
| Multi-objective (paper) | [DNG Reward](dng-reward.md) | Creativity + stability + diversity |
| Property optimization | [Predictor Reward](predictor-reward.md) | Train predictor, use as reward |

## Built-in Reward Components

All components are in [`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py):

| Component | Description | Required Metrics |
|-----------|-------------|------------------|
| `CustomReward` | User-defined reward function | None |
| `PredictorReward` | Property prediction from trained predictor | None |
| `CreativityReward` | Rewards unique and novel structures | `unique`, `novel` |
| `EnergyReward` | Penalizes high energy above convex hull | `e_above_hull` |
| `StructureDiversityReward` | Rewards diverse crystal geometries (MMD) | `structure_diversity` |
| `CompositionDiversityReward` | Rewards diverse chemical compositions (MMD) | `composition_diversity` |

## RewardComponent Base Class

All reward components inherit from `RewardComponent`:

```python
class RewardComponent(ABC, torch.nn.Module):
    def __init__(
        self,
        weight: float = 1.0,          # Relative importance
        normalize_fn: str | None = None,  # Normalization strategy
        eps: float = 1e-4,            # Numerical stability
    ):
        ...

    @abstractmethod
    def compute(self, **kwargs) -> torch.Tensor:
        """Compute raw reward values."""
        pass

    def forward(self, **kwargs) -> torch.Tensor:
        """Compute, normalize, and weight the reward."""
        rewards = self.compute(**kwargs)
        if self.normalize_fn:
            rewards = self._normalize(rewards)
        return rewards * self.weight
```

### Available kwargs in `compute()`

| Argument | Type | Description |
|----------|------|-------------|
| `gen_structures` | `list[Structure]` | Generated pymatgen Structure objects |
| `batch_gen` | `CrystalBatch` | Batched tensor representation |
| `metrics_obj` | `Metrics` | Pre-computed metrics (if `required_metrics` is set) |
| `device` | `torch.device` | Current device |

## Normalization Options

Each component can apply normalization via `normalize_fn`:

| Option | Formula | Use Case |
|--------|---------|----------|
| `norm` | `(x - min) / (max - min)` | Scale to [0, 1] |
| `std` | `(x - mean) / std` | Zero mean, unit variance |
| `subtract_mean` | `x - mean` | Center around zero |
| `clip` | `clamp(x, -1, 1)` | Bound extreme values |
| `null` | No change | Already normalized (e.g., CreativityReward) |

## ReinforceReward Aggregation

The `ReinforceReward` class (see [`src/rl_module/reward.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/reward.py)) combines multiple components:

```yaml
reward_fn:
  _target_: src.rl_module.reward.ReinforceReward
  normalize_fn: std           # Global normalization after combining
  eps: 1e-4
  reference_dataset: mp-20    # For novelty/uniqueness metrics
  components:
    - _target_: src.rl_module.components.CreativityReward
      weight: 1.0 # Weight for this component (default 1.0)
      normalize_fn: null # Component normalization
    - _target_: src.rl_module.components.EnergyReward
      weight: 0.5
      normalize_fn: norm # Component normalization
```

### How Rewards Are Combined

1. Each component computes its reward
2. Component-level normalization is applied (if specified)
3. Rewards are multiplied by weights
4. All weighted rewards are summed
5. Global normalization is applied (if specified)

:::{note}
**Global vs Component Normalization**: `ReinforceReward.normalize_fn` applies **global normalization** to the final combined reward, while each component's `normalize_fn` applies **per-component normalization** before weighting. We recommend `std` for global normalization and `norm` or `null` for component normalization.
:::

## Component Details

### CustomReward

Placeholder for user-defined logic. Modify directly in [`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py):

```python
class CustomReward(RewardComponent):
    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        # Your custom logic here
        rewards = [your_function(s) for s in gen_structures]
        return torch.as_tensor(rewards)
```

### CreativityReward

Rewards structures that are both unique (not duplicated in batch) and novel (not in training set):

- Returns 1.0 if unique AND novel
- Returns 0.0 if not unique AND not novel
- Uses AMD distance for edge cases

### EnergyReward

Penalizes high energy above the convex hull:

- Uses MACE-torch for energy calculations
- Clips energy to [0, 1] eV/atom
- Returns negative energy (lower is better)

### StructureDiversityReward

Encourages diverse crystal geometries using Maximum Mean Discrepancy (MMD):

- Compares generated structures to reference distribution
- Rewards structures that differ from existing ones
- Uses polynomial kernel for smooth gradients

### CompositionDiversityReward

Encourages diverse chemical compositions using MMD:

- Compares element distributions to reference
- Rewards exploration of chemical space

### PredictorReward

Uses a trained predictor as surrogate model:

```yaml
- _target_: src.rl_module.components.PredictorReward
  weight: 1.0
  predictor:
    _target_: src.vae_module.predictor_module.PredictorModule.load_from_checkpoint
    checkpoint_path: "ckpts/predictor.ckpt"
    map_location: "cpu"
  target_name: dft_band_gap  # Must match predictor's target_conditions key
  target_value: 3.0    # Optional: optimize toward this value
  clip_min: 0.0        # Optional: bound predictions
```

- If `target_value` is set: Returns `-(pred - target)²`
- If `target_value` is None: Returns `pred` (maximize)
- **Important**: `target_name` must match exactly with the key in predictor's `target_conditions`

## RL Configuration

Configure RL training behavior via `rl_module.rl_configs` (see [`configs/rl_module/rl_module.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/rl_module/rl_module.yaml)):

```yaml
rl_module:
  rl_configs:
    clip_ratio: 0.001
    kl_weight: 1.0
    entropy_weight: 1e-5
    num_group_samples: 64
    group_reward_norm: true
    num_inner_batch: 2
```

### Parameter Details

| Parameter | Default | Effect |
|-----------|---------|--------|
| `clip_ratio` | 0.001 | PPO clipping ratio. ↑ = larger policy updates (faster but unstable), ↓ = conservative updates (stable but slow) |
| `kl_weight` | 1.0 | KL divergence penalty. ↑ = stays closer to original policy, ↓ = allows more deviation |
| `entropy_weight` | 1e-5 | Entropy bonus. ↑ = more exploration/diversity, ↓ = more exploitation |
| `num_group_samples` | 1 | Samples per group for GRPO. ↑ = stable gradients (slow), ↓ = noisy gradients (fast) |
| `group_reward_norm` | false | Group reward normalization. `true` = GRPO (relative ranking), `false` = REINFORCE (absolute reward) |
| `num_inner_batch` | 2 | Gradient accumulation steps. ↑ = larger effective batch size |

:::{note}
**GRPO vs REINFORCE**: Set `num_group_samples >= 32` and `group_reward_norm: true` for GRPO. Default config uses REINFORCE (`num_group_samples: 1`, `group_reward_norm: false`).
:::


### Choosing a Starting Checkpoint

When starting RL training, you can choose between two LDM checkpoint options:

| Checkpoint | Description | Advantages | Disadvantages |
|------------|-------------|------------|---------------|
| `${hub:mp_20_ldm_base}` | Base LDM trained on likelihood | Broader chemical space exploration | No guarantee of high Msun structures |
| `${hub:mp_20_ldm_rl_dng}` | RL-finetuned LDM (DNG reward) | Guarantees high Msun structures | Narrower chemical space |

:::{tip}
**Recommended**: Start with `${hub:mp_20_ldm_base}` as your baseline. While `ldm_rl_dng` guarantees high Msun (match with training set), it may have learned a narrower chemical space. Starting from the base checkpoint allows your custom reward to explore more diverse material compositions.
:::

Example configuration:
```yaml
rl_module:
  ldm_ckpt_path: ${hub:mp_20_ldm_base}  # Recommended: broader exploration
  vae_ckpt_path: ${hub:mp_20_vae}
```

## Tutorials

- [Atomic Density](atomic-density.md) - Simple custom reward example
- [DNG Reward](dng-reward.md) - Paper's multi-objective configuration
- [Predictor Reward](predictor-reward.md) - Property optimization workflow
