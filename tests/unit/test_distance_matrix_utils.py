"""Unit tests for distance matrix utilities."""

import pytest
import torch
from pymatgen.core import Lattice, Structure

from src.data.dataset_util import pmg_structure_to_pyg_data
from src.data.schema import CrystalBatch
from src.utils.distance_matrix import compute_pbc_distance_matrix_batch


@pytest.mark.unit
def test_pbc_distance_matrix_simple_cubic():
    """Test distance matrix computation for simple cubic structure."""
    # Create simple cubic structure: 2 atoms
    lattice = Lattice.cubic(5.0)
    structure = Structure(
        lattice,
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    )

    # Convert to CrystalBatch
    data = pmg_structure_to_pyg_data(structure)
    batch = CrystalBatch.collate([data])

    # Compute distance matrix
    dist_matrices, masks = compute_pbc_distance_matrix_batch(batch)

    # Check shapes
    assert len(dist_matrices) == 1
    assert len(masks) == 1
    assert dist_matrices[0].shape == (2, 2)
    assert masks[0].shape == (2, 2)

    # Check diagonal is zero
    assert torch.allclose(torch.diag(dist_matrices[0]), torch.zeros(2), atol=1e-6)

    # Check symmetry
    assert torch.allclose(dist_matrices[0], dist_matrices[0].T, atol=1e-6)

    # Check expected distance (2.5 Å)
    expected_dist = 2.5
    assert torch.allclose(
        dist_matrices[0][0, 1],
        torch.tensor(expected_dist),
        atol=1e-3
    )
    assert torch.allclose(
        dist_matrices[0][1, 0],
        torch.tensor(expected_dist),
        atol=1e-3
    )


@pytest.mark.unit
def test_pbc_distance_matrix_periodic_image():
    """Test PBC handling: atoms near cell boundary."""
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Fe", "Fe"],
        [[0.01, 0.0, 0.0], [0.99, 0.0, 0.0]]  # Close via PBC
    )

    data = pmg_structure_to_pyg_data(structure)
    batch = CrystalBatch.collate([data])

    dist_matrices, _ = compute_pbc_distance_matrix_batch(batch)

    # Distance should be ~0.2 Å (via PBC), not ~9.8 Å
    assert dist_matrices[0][0, 1] < 1.0  # Much less than 10 Å
    expected_dist = 0.2  # 10 Å * (0.99 - 0.01 - 1.0)
    assert torch.allclose(
        dist_matrices[0][0, 1],
        torch.tensor(expected_dist),
        atol=0.1
    )


@pytest.mark.unit
def test_pbc_distance_matrix_batched():
    """Test distance matrix computation for batched structures."""
    # Create two different structures
    lattice1 = Lattice.cubic(5.0)
    structure1 = Structure(
        lattice1,
        ["Fe", "Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
    )

    lattice2 = Lattice.cubic(6.0)
    structure2 = Structure(
        lattice2,
        ["Cu", "Cu"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
    )

    # Convert to CrystalBatch
    data1 = pmg_structure_to_pyg_data(structure1)
    data2 = pmg_structure_to_pyg_data(structure2)
    batch = CrystalBatch.collate([data1, data2])

    # Compute distance matrices
    dist_matrices, masks = compute_pbc_distance_matrix_batch(batch)

    # Check we get two matrices
    assert len(dist_matrices) == 2
    assert len(masks) == 2

    # Check shapes
    assert dist_matrices[0].shape == (3, 3)  # First structure has 3 atoms
    assert dist_matrices[1].shape == (2, 2)  # Second structure has 2 atoms

    # Check diagonals are zero
    assert torch.allclose(torch.diag(dist_matrices[0]), torch.zeros(3), atol=1e-6)
    assert torch.allclose(torch.diag(dist_matrices[1]), torch.zeros(2), atol=1e-6)

    # Check symmetry
    assert torch.allclose(dist_matrices[0], dist_matrices[0].T, atol=1e-6)
    assert torch.allclose(dist_matrices[1], dist_matrices[1].T, atol=1e-6)


@pytest.mark.unit
def test_pbc_distance_matrix_non_cubic():
    """Test distance matrix computation for non-cubic lattice."""
    # Create orthorhombic lattice
    lattice = Lattice.orthorhombic(5.0, 6.0, 7.0)
    structure = Structure(
        lattice,
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    )

    data = pmg_structure_to_pyg_data(structure)
    batch = CrystalBatch.collate([data])

    dist_matrices, _ = compute_pbc_distance_matrix_batch(batch)

    # Check shape
    assert dist_matrices[0].shape == (2, 2)

    # Expected distance: 0.5 * 5.0 = 2.5 Å
    expected_dist = 2.5
    assert torch.allclose(
        dist_matrices[0][0, 1],
        torch.tensor(expected_dist),
        atol=1e-3
    )
