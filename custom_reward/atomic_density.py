"""Atomic density reward for RL training."""

import torch
from pymatgen.core import Structure

from src.rl_module.components import RewardComponent


class AtomicDensityReward(RewardComponent):
    """Reward based on atomic density (mass per volume)."""

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        rewards = []
        for structure in gen_structures:
            rewards.append(structure.density)
        return torch.as_tensor(rewards)
