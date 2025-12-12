# RL Module

The Reinforcement Learning module ([`src/rl_module/`](https://github.com/hspark1212/chemeleon2/tree/main/src/rl_module)) fine-tunes the LDM using Group Relative Policy Optimization (GRPO).

## Algorithm Overview

GRPO optimizes the LDM policy to maximize expected rewards:

$$\mathcal{L}_{GRPO} = -\mathbb{E}[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)] + \beta D_{KL} - \gamma H$$

Where:
- $r_t(\theta)$: Probability ratio between current and old policy
- $A_t$: Advantage (normalized group rewards)
- $\epsilon$: Clipping parameter
- $\beta$: KL penalty weight
- $\gamma$: Entropy weight
- $H$: Entropy (encourages high policy entropy)

## Key Classes

### RLModule

PyTorch Lightning module for RL fine-tuning ([`src/rl_module/rl_module.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/rl_module.py)):

```python
from src.rl_module import RLModule

# Load RL module from checkpoint
rl_module = RLModule.load_from_checkpoint(
    "path/to/rl.ckpt",
    weights_only=False,
)
```

**Key Methods:**
- `rollout(batch)` - Generates trajectories from LDM
- `compute_rewards(batch_gen)` - Evaluates generated structures
- `calculate_loss(...)` - Computes GRPO surrogate objective

### RewardComponent (Base Class)

Abstract base for all reward components ([`src/rl_module/components.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/components.py)):

```python
from src.rl_module.components import RewardComponent

class MyCustomReward(RewardComponent):
    def compute(self, gen_structures, **kwargs):
        # Return tensor of rewards for each structure
        # gen_structures: list[Structure]
        return rewards  # torch.Tensor
```

### Built-in Reward Components

| Component | Description |
|-----------|-------------|
| `CustomReward` | User-defined rewards |
| `PredictorReward` | Surrogate model predictions |
| `CreativityReward` | Unique + Novel structures |
| `EnergyReward` | Low energy above hull |
| `StructureDiversityReward` | MMD-based diversity |
| `CompositionDiversityReward` | Composition diversity |

### ReinforceReward

Aggregates multiple reward components ([`src/rl_module/reward.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/rl_module/reward.py)):

```python
from src.rl_module.reward import ReinforceReward
from src.rl_module.components import CreativityReward, EnergyReward

reward = ReinforceReward(
    components=[
        CreativityReward(weight=1.0),
        EnergyReward(weight=0.5),
    ],
    normalize_fn="std",
)
```

## Normalization Strategies

| Strategy | Description |
|----------|-------------|
| `std` | Standardize to zero mean, unit variance |
| `norm` | Min-max normalization to [0, 1] |
| `subtract_mean` | Subtract mean only |
| `clip` | Clip to specified range |

## Configuration

See [`configs/rl_module/`](https://github.com/hspark1212/chemeleon2/tree/main/configs/rl_module) for RL configurations:

```yaml
# configs/rl_module/rl_module.yaml (default)
_target_: src.rl_module.rl_module.RLModule
reward_fn:
  _target_: src.rl_module.reward.ReinforceReward
  normalize_fn: std
  components:
    - _target_: src.rl_module.components.CreativityReward
      weight: 1.0
    - _target_: src.rl_module.components.EnergyReward
      weight: 1.0
    - _target_: src.rl_module.components.StructureDiversityReward
      weight: 0.1
    - _target_: src.rl_module.components.CompositionDiversityReward
      weight: 1.0
rl_configs:
  clip_ratio: 0.001
  kl_weight: 1.0
  num_group_samples: 1
```

## Training

```bash
# De novo generation RL
python src/train_rl.py experiment=mp_20/rl_dng

# Band gap optimization RL
python src/train_rl.py experiment=alex_mp_20_bandgap/rl_bandgap
```

Training script: [`src/train_rl.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_rl.py)

See [Training Guide](../user-guide/training/index.md) and [Custom Rewards](../user-guide/rewards/index.md) for more details.
