"""Placeholder for a custom reward component."""

import torch
from pymatgen.core import Structure

from src.rl_module.components import RewardComponent


class CustomReward(RewardComponent):
    """Create your custom reward component."""

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        rewards = []
        for structure in gen_structures:
            rewards.append(0)  # Placeholder for custom reward logic
        return torch.as_tensor(rewards)
