"""Distance matrix computation utilities for VAE training."""

import torch

from src.data.schema import CrystalBatch


def compute_pbc_distance_matrix_batch(
    batch: CrystalBatch,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute PBC-aware distance matrices for batched structures.

    Uses pymatgen's Structure.distance_matrix which handles periodic
    boundary conditions via minimum image convention.

    Args:
        batch: CrystalBatch with structures

    Returns:
        distance_matrices: List of (N_i, N_i) distance matrices in Ångströms
        masks: List of (N_i, N_i) boolean masks (all True, for interface consistency)
    """
    # Convert batch to list of pymatgen structures
    structures = batch.to_structure()

    distance_matrices = []
    masks = []

    for structure in structures:
        # Compute PBC-aware distance matrix using pymatgen
        # Returns numpy array of shape (N, N) with distances in Ångströms
        dist_matrix_np = structure.distance_matrix

        # Convert to PyTorch tensor
        dist_matrix = torch.from_numpy(dist_matrix_np).float()

        # Create mask (all True for now, for interface consistency)
        mask = torch.ones_like(dist_matrix, dtype=torch.bool)

        distance_matrices.append(dist_matrix)
        masks.append(mask)

    return distance_matrices, masks
