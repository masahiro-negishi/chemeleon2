"""Shared pytest fixtures for chemeleon2 test suite.

This module provides common fixtures for baseline, contract, integration,
and unit tests. Fixtures include device detection, dummy crystal data,
and reproducibility helpers for PyTorch Lightning models.
"""

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def device() -> str:
    """Detect and return the available compute device (cuda/cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="function")
def seed_everything():
    """Set random seeds for reproducibility across numpy, torch, and Python."""

    def _seed(seed_value=42) -> None:
        import random

        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return _seed


@pytest.fixture(scope="function")
def dummy_crystal_batch(device):
    """Create a small dummy CrystalBatch for testing model forward passes.

    This fixture generates synthetic crystal structure data compatible with
    VAE, LDM, and RL modules. All tensors are initialized with valid values
    to avoid CUDA device assertions and NaN issues.
    Uses realistic atom count distributions from num_atom_distributions.
    """

    def _create_batch(batch_size=2, num_atom_distribution="mp-20"):
        """Generate dummy CrystalBatch with specified dimensions.

        Args:
            batch_size: Number of crystal structures in batch
            num_atom_distribution: Distribution name ("mp-20" or "mp-120")

        Returns:
            CrystalBatch object ready for model testing
        """
        from torch_geometric.data import Data

        from src.data.num_atom_distributions import NUM_ATOM_DISTRIBUTIONS
        from src.data.schema import CrystalBatch

        distribution = NUM_ATOM_DISTRIBUTIONS[num_atom_distribution]
        num_atoms = np.random.choice(
            list(distribution.keys()),
            p=list(distribution.values()),
            size=batch_size,
        ).tolist()

        # Generate properly initialized dummy data for each structure
        data_list = []
        for n in num_atoms:
            # Generate random atom types (1-99, avoiding 0)
            atom_types = torch.randint(1, 100, (n,), dtype=torch.long)

            # Generate random fractional coordinates (0-1)
            frac_coords = torch.rand((n, 3))

            # Generate random lattice (identity matrix with small perturbations)
            lattice = torch.eye(3).unsqueeze(0) + torch.randn((1, 3, 3)) * 0.1

            # Generate lattice parameters
            lengths = torch.rand((1, 3)) * 5 + 5  # Between 5-10 Angstroms
            lengths_scaled = lengths / (n ** (1 / 3))  # Scale by num_atoms^(1/3)
            angles = torch.ones((1, 3)) * (torch.pi / 2)  # 90 degrees
            angles_radians = angles.clone()

            # Calculate cartesian coordinates
            cart_coords = torch.einsum("bij,ni->nj", lattice, frac_coords)

            data_list.append(
                Data(
                    pos=cart_coords,
                    atom_types=atom_types,
                    frac_coords=frac_coords,
                    cart_coords=cart_coords,
                    lattices=lattice,
                    num_atoms=torch.as_tensor(n, dtype=torch.long),
                    lengths=lengths,
                    lengths_scaled=lengths_scaled,
                    angles=torch.rad2deg(angles),
                    angles_radians=angles_radians,
                    token_idx=torch.arange(n, dtype=torch.long),
                )
            )

        return CrystalBatch.from_data_list(data_list).to(device=device)

    return _create_batch


@pytest.fixture(scope="function")
def simple_encoder(device):
    """Create a minimal TransformerEncoder for testing."""
    from src.vae_module.encoders.transformer import TransformerEncoder

    return TransformerEncoder(
        max_num_elements=100,
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        dropout=0.0,
        num_layers=2,
    ).to(device)


@pytest.fixture(scope="function")
def simple_distance_decoder(device):
    """Create a minimal DistanceMatrixDecoder for testing."""
    from src.vae_module.decoders.distance_matrix import DistanceMatrixDecoder

    return DistanceMatrixDecoder(
        max_num_elements=100,
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        dropout=0.0,
        num_layers=2,
        atom_type_predict=True,
        distance_mlp_hidden_dims=[128, 64],
        classifier_hidden_dims=[128, 64],
        use_vectorized=True,
    ).to(device)


@pytest.fixture(scope="function")
def vae_distance_model(simple_encoder, simple_distance_decoder):
    """Create a minimal VAE with distance decoder for testing."""
    import torch
    from omegaconf import OmegaConf
    from pymatgen.analysis.structure_matcher import StructureMatcher

    from src.vae_module.vae_module import VAEModule

    model = VAEModule(
        encoder=simple_encoder,
        decoder=simple_distance_decoder,
        latent_dim=64,
        distance_threshold=10.0,
        loss_weights=OmegaConf.create(
            {
                "atom_types": 1.0,
                "lengths": 1.0,
                "angles": 1.0,
                "frac_coords": 0.0,
                "distance_regression": 10.0,
                "distance_classifier": 10.0,
                "kl": 0.01,
                "fa": 0.0,
            }
        ),
        augmentation=OmegaConf.create(
            {
                "translate": False,
                "rotate": False,
            }
        ),
        noise=OmegaConf.create(
            {
                "ratio": 0.0,
                "corruption_scale": 0.0,
            }
        ),
        atom_type_predict=True,
        structure_matcher=StructureMatcher(),
        optimizer=torch.optim.Adam,
        scheduler=None,
    )
    return model
