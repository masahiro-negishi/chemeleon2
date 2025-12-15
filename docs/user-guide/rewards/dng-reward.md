# Tutorial: DNG (De Novo Generation) Reward

Learn to configure the multi-objective reward used in the Chemeleon2 paper for generating novel, stable, and diverse crystal structures.

## Overview

The DNG reward combines four complementary objectives:

| Component | Purpose |
|-----------|---------|
| **CreativityReward** | Generate unique and novel structures |
| **EnergyReward** | Ensure thermodynamic stability |
| **StructureDiversityReward** | Explore varied structure prototypes |
| **CompositionDiversityReward** | Explore chemical composition space |

## Prerequisites

- **MP-20 reference dataset** - Required for evaluation metrics (novelty, diversity, etc.)

:::{important}
**Required**: Download the MP-20 reference dataset from Figshare to compute evaluation metrics. Without this, the DNG reward components (CreativityReward, StructureDiversityReward, CompositionDiversityReward) will not work properly.

See [Evaluation Guide - Prerequisites](../evaluation.md#prerequisites) for download instructions.
:::

## The DNG Configuration

Reference file: [`configs/custom_reward/rl_dng.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/custom_reward/rl_dng.yaml)

```yaml
# @package _global_
# GRPO for DNG on MP-20

data:
  batch_size: 5

rl_module:
  ldm_ckpt_path: ${hub:mp_20_ldm_base}
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
      - _target_: src.rl_module.components.CreativityReward
        weight: 1.0
        normalize_fn: null
      - _target_: src.rl_module.components.EnergyReward
        weight: 1.0
        normalize_fn: norm
      - _target_: src.rl_module.components.StructureDiversityReward
        weight: 0.1
        normalize_fn: norm
      - _target_: src.rl_module.components.CompositionDiversityReward
        weight: 1.0
        normalize_fn: norm

logger:
  wandb:
    name: rl_dng_grpo
```

:::{tip}
**Adjustable Weights**: You can customize the importance of each reward component by modifying the `weight` parameter. See [Weight Tuning Guide](#weight-tuning-guide) below for recommendations based on your optimization goals.
:::

:::{note}
The `${hub:...}` syntax automatically downloads pre-trained models from HuggingFace. See [Automatic Download from HuggingFace](../training/index.md#automatic-download-from-huggingface) for details.
:::

## Component Deep Dive

The DNG reward uses **built-in reward components** provided by Chemeleon2. For a complete list of available components, see [RL Module - Built-in Reward Components](../../architecture/rl-module.md#built-in-reward-components). These components are ready to use without additional implementation.

### [CreativityReward](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py#L92)

**Purpose:** Reward structures that are both unique (not duplicated in batch) and novel (not in training set).

**How it works:**

```python
for i, gen_structure in enumerate(gen_structures):
    u, v = metrics_results["unique"][i], metrics_results["novel"][i]
    if u and v:
        r = 1.0  # Fully creative: unique AND novel
    elif not u and not v:
        r = 0.0  # Not creative: duplicate of existing
    else:
        # Edge case: use AMD distance as continuous measure
        amds = structures_to_amd([gen_structure] + matching_refs, 100)
        dists = amd.AMD_cdist(amds, amds)[0]
        r = dists[dists > 0].min()
```

**Configuration:**
- `weight: 1.0` - Equal importance with other objectives
- `normalize_fn: null` - Already in [0, 1] range

### [EnergyReward](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py#L134)

**Purpose:** Penalize structures with high energy above the convex hull.

**How it works:**
- Computes formation energy using MACE-torch
- Compares to Materials Project convex hull
- Returns negative energy (minimization → maximization)

```python
r_energy = torch.as_tensor(metrics_results["e_above_hull"]).float()
r_energy = r_energy.nan_to_num(nan=1.0)  # Handle failed calculations
r_energy = r_energy.clamp(min=0.0, max=1.0)  # Clip to reasonable range
r_energy = r_energy * -1.0  # Negative for minimization
```

**Configuration:**
- `weight: 1.0` - Strong emphasis on stability
- `normalize_fn: norm` - Scale within batch

### [StructureDiversityReward](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py#L154)

**Purpose:** Encourage diverse crystal geometries using Maximum Mean Discrepancy (MMD).

**How it works:**
- Featurizes structures (lattice parameters, atomic positions)
- Computes MMD between generated batch and reference distribution
- Rewards structures that differ from existing patterns

**Configuration:**
- `weight: 0.1` - Lower weight prevents over-diversification
- `normalize_fn: norm` - Scale within batch

**Why lower weight?**
Too much structure diversity can lead to:
- Physically unrealistic geometries
- Sacrificing stability for novelty

### [CompositionDiversityReward](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py#L185)

**Purpose:** Encourage exploration of chemical composition space.

**How it works:**
- Extracts element-wise composition features
- Computes MMD between generated and reference compositions
- Rewards deviating from common compositions

**Configuration:**
- `weight: 1.0` - Strong emphasis on chemical diversity
- `normalize_fn: norm` - Scale within batch

## Running DNG Training

```bash
# Standard DNG training (src/train_rl.py)
python src/train_rl.py custom_reward=rl_dng

# With custom hyperparameters
python src/train_rl.py custom_reward=rl_dng \
    rl_module.rl_configs.num_group_samples=128 \
    trainer.max_steps=2000

# Override checkpoint paths
python src/train_rl.py custom_reward=rl_dng \
    rl_module.ldm_ckpt_path=ckpts/my_ldm.ckpt
```

## Monitoring Training

:::{tip}
**Example training run**: See [this W&B run](https://wandb.ai/hspark1212/chemeleon2_project/runs/qlui3icn) for DNG reward training examples.
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

(weight-tuning-guide)=
## Weight Tuning Guide

Adjust weights based on your priorities:

| Priority | CreativityReward | EnergyReward | StructureDiversity | CompositionDiversity |
|----------|------------------|--------------|--------------------|-----------------------|
| More novelty | ↑ 1.5 | ↓ 0.5 | 0.1 | 1.0 |
| More stability | 0.5 | ↑ 2.0 | 0.1 | 0.5 |
| More diversity | 1.0 | 0.5 | ↑ 0.5 | ↑ 1.5 |
| Balanced (default) | 1.0 | 1.0 | 0.1 | 1.0 |

```yaml
# Example: Prioritize stability
components:
  - _target_: src.rl_module.components.CreativityReward
    weight: 0.5
  - _target_: src.rl_module.components.EnergyReward
    weight: 2.0
    normalize_fn: norm
  - _target_: src.rl_module.components.StructureDiversityReward
    weight: 0.1
    normalize_fn: norm
  - _target_: src.rl_module.components.CompositionDiversityReward
    weight: 0.5
    normalize_fn: norm
```

## Generating and Evaluating Samples

After training your DNG model, you can generate 10,000 structures and evaluate them against reference datasets to assess quality.

### Generate Samples

Generate crystal structures using the trained RL model:

```bash
# Generate 10000 samples with batch size 2000
python src/sample.py \
    --ldm_ckpt_path=logs/train_rl/runs/<your-run>/checkpoints/last.ckpt \
    --num_samples=10000 \
    --batch_size=2000 \
    --output_dir=outputs/dng_samples
```

### Evaluate Samples

The evaluation computes several quality metrics:

| Metric | Description |
|--------|-------------|
| **Unique** | Structures not duplicated within generated set |
| **Novel** | Structures not found in reference dataset |
| **E Above Hull** | Energy above convex hull (stability measure) |
| **Metastable/Stable** | Thermodynamically viable structures |
| **Composition Validity** | Chemically valid compositions (via SMACT) |
| **Structure Diversity** | Inverse Fréchet distance for structure embeddings |
| **Composition Diversity** | Inverse Fréchet distance for composition embeddings |

```bash
python src/evaluate.py \
    --model_path=logs/train_rl/runs/<your-run>/checkpoints/last.ckpt \
    --structure_path=outputs/dng_samples \
    --num_samples=10000 \
    --batch_size=2000 \
    --output_file=outputs/dng_samples/results.csv
```

:::{tip}
See [Evaluation Guide](../evaluation.md) for detailed usage and Python API examples.
:::

## Summary

The DNG reward configuration:
1. **Balances multiple objectives** for well-rounded generation
2. **Prevents mode collapse** with diversity rewards
3. **Ensures physical validity** with energy penalty
4. **Encourages exploration** with creativity bonus

## Next Steps

- [Predictor Reward](predictor-reward.md) - Property-targeted optimization
- [Atomic Density](atomic-density.md) - Simple custom reward example
