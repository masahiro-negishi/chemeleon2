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

    Args:
        predictor: Trained PredictorModule for band gap prediction
        target_value: Target band gap value in eV (default: 3.0)
        clip_min: Minimum allowed band gap value (default: 0.0)
        clip_max: Maximum allowed band gap value (default: None)
        **kwargs: Additional arguments passed to RewardComponent (weight, normalize_fn, eps)

    Example:
        >>> predictor = PredictorModule.load_from_checkpoint(
        ...     "ckpts/mp_20/predictor/predictor_band_gap.ckpt",
        ...     weights_only=False
        ... )
        >>> reward = BandGapPredictorReward(
        ...     predictor=predictor,
        ...     target_value=3.0,
        ...     clip_min=0.0,
        ... )
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
        """Compute reward based on predicted band gap values.

        Args:
            batch_gen: Generated crystal structures in batch format
            **kwargs: Additional keyword arguments

        Returns:
            Reward tensor for each structure in the batch
        """
        device = self.predictor.device
        batch_gen = batch_gen.to(device)

        # Get predictions from the predictor
        pred = self.predictor.predict(batch_gen)
        pred_val = pred[self.target_name].clamp(min=self.clip_min, max=self.clip_max)

        # Compute reward based on target value
        if self.target_value is not None:
            # Reward is negative squared error if target is specified
            reward = -((pred_val - self.target_value) ** 2)
        else:
            # Otherwise maximize the predicted value
            reward = pred_val

        return reward
