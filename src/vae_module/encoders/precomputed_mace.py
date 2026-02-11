"""Wrapper encoder for pre-computed MACE embeddings."""

import torch
from torch import Tensor, nn

from src.data.schema import CrystalBatch


class PrecomputedMACEEncoder(nn.Module):
    """Encoder that returns pre-computed MACE embeddings.

    Expects CrystalBatch to have 'mace_embeddings' attribute attached
    by MPDataset. Simply reformats to match encoder output contract.

    This wrapper is used when training with pre-computed embeddings to avoid
    redundant forward passes through the frozen MACE model. The embeddings
    are automatically pre-computed on first use by DataModule using
    src/utils/precompute.py and stored in mace_embeddings.h5.

    Args:
        hidden_dim: Dimension of embeddings (must match pre-computed).
            Default 512 matches MACE mh-1 final layer invariants.
            For other models, check MACEEncoder.hidden_dim.
        max_num_elements: For config compatibility (unused, but kept for
            consistency with other encoders).
    """

    def __init__(self, hidden_dim: int = 512, max_num_elements: int = 100):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._max_num_elements = max_num_elements

    @property
    def hidden_dim(self) -> int:
        """Return the hidden dimension of the embeddings."""
        return self._hidden_dim

    @property
    def max_num_elements(self) -> int:
        """Return max number of elements (for config compatibility)."""
        return self._max_num_elements

    def forward(self, batch: CrystalBatch) -> dict[str, Tensor]:
        """Return pre-computed embeddings.

        Args:
            batch: CrystalBatch with mace_embeddings attribute.

        Returns:
            Dict with keys "x", "num_atoms", "batch", "token_idx".

        Raises:
            ValueError: If batch does not have mace_embeddings attribute.
        """
        if not hasattr(batch, "mace_embeddings"):
            msg = (
                "CrystalBatch must have 'mace_embeddings' attribute. "
                "Set data.mace_embeddings=true in config. "
                "Embeddings are auto-computed on first use by DataModule."
            )
            raise ValueError(msg)

        return {
            "x": batch.mace_embeddings,
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }
