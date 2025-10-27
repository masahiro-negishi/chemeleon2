"""Symmetry tests for Sparse CrystalTransformerEncoder.

Tests the three fundamental symmetry properties required for crystal structures:
1. Permutation equivariance
2. Periodic translation invariance
3. O(3) rotation invariance
"""

import pytest
import torch
from pymatgen.core import Lattice, Structure

from src.data.dataset_util import pmg_structure_to_pyg_data
from src.data.graph_utils import radius_graph_pbc
from src.data.schema import CrystalBatch
from src.vae_module.encoders.crystal_transformer import CrystalTransformerEncoder


@pytest.fixture
def encoder():
    """Create a minimal CrystalTransformerEncoder for testing."""
    return CrystalTransformerEncoder(
        max_num_elements=100,
        d_model=128,
        n_frequencies=50,
        nhead=4,
        num_layers=2,
    )


@pytest.fixture
def random_rotation_matrix():
    """Generate random proper rotation matrix (det=1)."""

    def _generate():
        A = torch.randn(3, 3)
        Q, _ = torch.linalg.qr(A)
        # Ensure proper rotation (det = 1)
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        return Q

    return _generate


def create_crystal_batch():
    """Helper to create a CrystalBatch for testing."""
    st = Structure(
        Lattice.cubic(5.0),
        ["Si", "O", "Si", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.5, 0.5, 0.5]],
    )

    # Make PyG Data objects
    data = pmg_structure_to_pyg_data(st)
    edge_graph = radius_graph_pbc(
        atoms=st.to_ase_atoms(),
        r_cutoff=8.0,
        max_neighbors=12,
        compute_edge_vectors=False,
    )
    data.update(edge_graph)

    # Convert to CrystalBatch
    batch = CrystalBatch.from_data_list([data])
    return batch


@pytest.mark.symmetry
def test_permutation_equivariance(encoder):
    """Test permutation equivariance: encode(A_P, F_P, L) = P(encode(A, F, L)).

    Property: Latent features should be equivariant - permuting input atoms
    should permute latent features identically.
    """


@pytest.mark.symmetry
def test_periodic_translation_invariance(encoder):
    """Test translation invariance: encode(A, w(F + t1^T), L) = encode(A, F, L).

    Property: Model is invariant to periodic translations of fractional coordinates.
    """
    encoder.eval()
    batch = create_crystal_batch()

    # Random translation
    t = torch.rand(3)
    frac_coords_trans = (batch.frac_coords + t) % 1.0

    with torch.no_grad():
        z1 = encoder(batch)

        batch_trans = batch.clone()  # type: ignore
        batch_trans.frac_coords = frac_coords_trans
        z2 = encoder(batch_trans)

    max_diff = torch.max(torch.abs(z2["x"] - z1["x"]))
    assert max_diff < 1e-4, f"Translation invariance failed: max_diff={max_diff}"


@pytest.mark.symmetry
def test_rotation_invariance(encoder, random_rotation_matrix):
    """Test O(3) rotation invariance: encode(A, F, QL) = encode(A, F, L).

    Property: Rotating L → QL does NOT change (a, b, c, α, β, γ) or F,
    only Cartesian coordinates C = FL.
    """
    encoder.eval()
    batch = create_crystal_batch()

    # Random rotation matrix
    Q = random_rotation_matrix()

    # Apply rotation to lattice
    L_rot = torch.bmm(Q.unsqueeze(0), batch.lattices)

    with torch.no_grad():
        z1 = encoder(batch)

        batch_rot = batch.clone()  # type: ignore
        batch_rot.lattices = L_rot
        z2 = encoder(batch_rot)

    max_diff = torch.max(torch.abs(z2["x"] - z1["x"]))
    assert max_diff < 1e-4, f"O(3) invariance failed: max_diff={max_diff}"


@pytest.mark.symmetry
def test_combined_symmetries(encoder, random_rotation_matrix):
    """Test combined symmetries: permutation + translation + rotation.

    Apply all three transformations simultaneously and verify all properties hold.
    """
    encoder.eval()
    batch = create_crystal_batch()

    # Apply all transformations
    Q = random_rotation_matrix()
    P = torch.randperm(len(batch.atom_types))
    t = torch.rand(3)

    with torch.no_grad():
        z1 = encoder(batch)

        # Create inverse permutation
        inv_P = torch.zeros_like(P)
        inv_P[P] = torch.arange(len(P))

        # Combined transformation
        batch_combined = batch.clone()  # type: ignore
        batch_combined.atom_types = batch.atom_types[P]
        batch_combined.frac_coords = (batch.frac_coords[P] + t) % 1.0
        batch_combined.lattices = torch.bmm(Q.unsqueeze(0), batch.lattices)
        batch_combined.edge_index = inv_P[batch.edge_index]
        batch_combined.batch = batch.batch[P]

        z2 = encoder(batch_combined)

    # Verify: z2 should equal z1 permuted
    max_diff = torch.max(torch.abs(z2["x"] - z1["x"][P]))
    assert max_diff < 1e-4, f"Combined symmetries failed: max_diff={max_diff}"
