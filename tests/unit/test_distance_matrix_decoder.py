"""Unit tests for distance matrix decoder."""

import pytest
import torch

from src.vae_module.decoders.distance_matrix import DistanceMatrixDecoder


@pytest.mark.unit
def test_distance_matrix_decoder_init():
    """Test decoder initialization."""
    decoder = DistanceMatrixDecoder(
        d_model=256,
        num_layers=2,
        max_num_elements=100,
    )

    assert decoder.d_model == 256
    assert decoder.num_layers == 2
    assert decoder.max_num_elements == 100
    assert decoder.hidden_dim == 256


@pytest.mark.unit
def test_distance_matrix_decoder_output_shape():
    """Test decoder outputs correct shapes."""
    decoder = DistanceMatrixDecoder(
        d_model=256,
        num_layers=2,
        atom_type_predict=True,
        max_num_elements=100,
    )

    # Mock input (10 atoms: 4 in first structure, 6 in second)
    encoded_batch = {
        "x": torch.randn(10, 256),
        "num_atoms": torch.tensor([4, 6]),
        "batch": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        "token_idx": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5]),
    }

    output = decoder(encoded_batch)

    # Check output keys
    assert "distance_matrix" in output
    assert "atom_types" in output
    assert "lattices" in output
    assert "lengths" in output
    assert "angles" in output
    assert "frac_coords" in output

    # Check distance matrices
    assert len(output["distance_matrix"]) == 2  # Two structures
    assert output["distance_matrix"][0].shape == (4, 4)
    assert output["distance_matrix"][1].shape == (6, 6)

    # Check other outputs
    assert output["atom_types"].shape == (10, 100)
    assert output["lengths"].shape == (2, 3)
    assert output["angles"].shape == (2, 3)
    assert output["frac_coords"].shape == (10, 3)

    # Check dummy fractional coordinates are zeros
    assert torch.allclose(
        output["frac_coords"], torch.zeros_like(output["frac_coords"])
    )


@pytest.mark.unit
def test_distance_matrix_symmetry():
    """Test predicted distance matrices are symmetric."""
    decoder = DistanceMatrixDecoder(d_model=256, num_layers=2)

    encoded_batch = {
        "x": torch.randn(5, 256),
        "num_atoms": torch.tensor([5]),
        "batch": torch.zeros(5, dtype=torch.long),
        "token_idx": torch.arange(5),
    }

    output = decoder(encoded_batch)
    D = output["distance_matrix"][0]

    # Check symmetry: D[i,j] == D[j,i]
    assert torch.allclose(D, D.T, atol=1e-6)


@pytest.mark.unit
def test_distance_matrix_positive():
    """Test predicted distances are non-negative."""
    decoder = DistanceMatrixDecoder(d_model=256, num_layers=2)

    encoded_batch = {
        "x": torch.randn(5, 256),
        "num_atoms": torch.tensor([5]),
        "batch": torch.zeros(5, dtype=torch.long),
        "token_idx": torch.arange(5),
    }

    output = decoder(encoded_batch)
    D = output["distance_matrix"][0]

    # Check all distances are non-negative
    assert (D >= 0).all()


@pytest.mark.unit
def test_distance_matrix_zero_diagonal():
    """Test diagonal is zero (self-distance)."""
    decoder = DistanceMatrixDecoder(d_model=256, num_layers=2)

    encoded_batch = {
        "x": torch.randn(5, 256),
        "num_atoms": torch.tensor([5]),
        "batch": torch.zeros(5, dtype=torch.long),
        "token_idx": torch.arange(5),
    }

    output = decoder(encoded_batch)
    D = output["distance_matrix"][0]

    # Check diagonal is zero
    assert torch.allclose(torch.diag(D), torch.zeros(5), atol=1e-6)


@pytest.mark.unit
def test_distance_matrix_decoder_no_atom_type_predict():
    """Test decoder with atom_type_predict=False."""
    decoder = DistanceMatrixDecoder(
        d_model=256,
        num_layers=2,
        atom_type_predict=False,
    )

    encoded_batch = {
        "x": torch.randn(5, 256),
        "num_atoms": torch.tensor([5]),
        "batch": torch.zeros(5, dtype=torch.long),
        "token_idx": torch.arange(5),
    }

    output = decoder(encoded_batch)

    # atom_types should be None when not predicting
    assert output["atom_types"] is None


@pytest.mark.unit
def test_distance_matrix_decoder_multiple_structures():
    """Test decoder with multiple structures of different sizes."""
    decoder = DistanceMatrixDecoder(d_model=128, num_layers=2)

    # Three structures with 2, 3, and 5 atoms
    num_atoms_list = [2, 3, 5]
    total_atoms = sum(num_atoms_list)

    batch_idx = []
    token_idx = []
    for i, n in enumerate(num_atoms_list):
        batch_idx.extend([i] * n)
        token_idx.extend(list(range(n)))

    encoded_batch = {
        "x": torch.randn(total_atoms, 128),
        "num_atoms": torch.tensor(num_atoms_list),
        "batch": torch.tensor(batch_idx),
        "token_idx": torch.tensor(token_idx),
    }

    output = decoder(encoded_batch)

    # Check we get three distance matrices
    assert len(output["distance_matrix"]) == 3
    assert output["distance_matrix"][0].shape == (2, 2)
    assert output["distance_matrix"][1].shape == (3, 3)
    assert output["distance_matrix"][2].shape == (5, 5)

    # Check all constraints for each matrix
    for D in output["distance_matrix"]:
        # Symmetry
        assert torch.allclose(D, D.T, atol=1e-6)
        # Non-negative
        assert (D >= 0).all()
        # Zero diagonal
        assert torch.allclose(torch.diag(D), torch.zeros(D.shape[0]), atol=1e-6)


@pytest.mark.unit
def test_distance_matrix_decoder_custom_mlp():
    """Test decoder with custom MLP hidden dimensions."""
    decoder = DistanceMatrixDecoder(
        d_model=256,
        num_layers=2,
        distance_mlp_hidden_dims=[128, 64, 32],
    )

    encoded_batch = {
        "x": torch.randn(5, 256),
        "num_atoms": torch.tensor([5]),
        "batch": torch.zeros(5, dtype=torch.long),
        "token_idx": torch.arange(5),
    }

    output = decoder(encoded_batch)

    # Should still produce valid output
    assert "distance_matrix" in output
    assert output["distance_matrix"][0].shape == (5, 5)

    # Check constraints
    D = output["distance_matrix"][0]
    assert torch.allclose(D, D.T, atol=1e-6)
    assert (D >= 0).all()
    assert torch.allclose(torch.diag(D), torch.zeros(5), atol=1e-6)


@pytest.mark.unit
def test_vectorized_vs_sequential_equivalence():
    """Verify vectorized produces identical results to sequential."""
    torch.manual_seed(42)  # For reproducibility

    # Test various batch sizes and graph sizes
    test_cases = [
        [4, 8, 12, 16, 20],  # Mixed sizes
        [10] * 50,  # Many small
        [20] * 10,  # Few large
        [1, 2, 3, 4, 5],  # Very small
    ]

    for num_atoms_list in test_cases:
        # Build encoded batch
        total_atoms = sum(num_atoms_list)
        batch_idx = []
        token_idx = []
        for i, n in enumerate(num_atoms_list):
            batch_idx.extend([i] * n)
            token_idx.extend(list(range(n)))

        x = torch.randn(total_atoms, 256)
        num_atoms = torch.tensor(num_atoms_list)
        batch = torch.tensor(batch_idx)

        # Create decoder (use same instance for fair comparison)
        decoder = DistanceMatrixDecoder(
            d_model=256,
            num_layers=2,
            use_vectorized=False,  # Will manually test both paths
        )
        decoder.eval()  # Disable dropout

        # Compare outputs
        with torch.no_grad():
            seq_output, seq_logits = decoder._predict_distance_matrices_sequential(
                x, batch, num_atoms
            )
            vec_output, vec_logits = decoder._predict_distance_matrices_vectorized(
                x, batch, num_atoms
            )

        assert len(seq_output) == len(vec_output), (
            f"Length mismatch: {len(seq_output)} vs {len(vec_output)}"
        )
        for i, (D_seq, D_vec) in enumerate(zip(seq_output, vec_output, strict=False)):
            assert torch.allclose(D_seq, D_vec, atol=1e-5, rtol=1e-4), (
                f"Mismatch in graph {i} (size {num_atoms_list[i]}): "
                f"max_diff={torch.abs(D_seq - D_vec).max():.6f}"
            )


@pytest.mark.unit
def test_vectorized_edge_cases():
    """Test edge cases for vectorized implementation."""
    decoder = DistanceMatrixDecoder(d_model=128, num_layers=2)
    decoder.eval()

    # Empty batch
    with torch.no_grad():
        result, logits = decoder._predict_distance_matrices_vectorized(
            torch.empty(0, 128),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
    assert len(result) == 0, "Empty batch should return empty list"

    # Single atom
    with torch.no_grad():
        result, logits = decoder._predict_distance_matrices_vectorized(
            torch.randn(1, 128),
            torch.tensor([0]),
            torch.tensor([1]),
        )
    assert len(result) == 1, "Single atom should return one matrix"
    assert result[0].shape == (1, 1), f"Expected (1,1), got {result[0].shape}"
    assert torch.allclose(result[0], torch.zeros(1, 1), atol=1e-6), (
        "Single atom should have zero self-distance"
    )

    # Single graph with multiple atoms
    with torch.no_grad():
        result, logits = decoder._predict_distance_matrices_vectorized(
            torch.randn(5, 128),
            torch.zeros(5, dtype=torch.long),
            torch.tensor([5]),
        )
    assert len(result) == 1, "Single graph should return one matrix"
    assert result[0].shape == (5, 5), f"Expected (5,5), got {result[0].shape}"
    # Check symmetry
    assert torch.allclose(result[0], result[0].T, atol=1e-6), (
        "Matrix should be symmetric"
    )
    # Check zero diagonal
    assert torch.allclose(torch.diag(result[0]), torch.zeros(5), atol=1e-6), (
        "Diagonal should be zero"
    )


@pytest.mark.unit
def test_vectorized_memory_fallback():
    """Test that memory guard triggers fallback correctly."""
    decoder = DistanceMatrixDecoder(d_model=128, num_layers=2, use_vectorized=True)
    decoder.eval()

    # Create a batch that exceeds the 100K pairs threshold
    # With 20 atoms per graph: 20*21/2 = 210 pairs
    # Need ~477 graphs to exceed 100K pairs
    num_graphs = 500
    num_atoms_list = [20] * num_graphs
    total_atoms = sum(num_atoms_list)

    batch_idx = []
    token_idx = []
    for i, n in enumerate(num_atoms_list):
        batch_idx.extend([i] * n)
        token_idx.extend(list(range(n)))

    x = torch.randn(total_atoms, 128)
    num_atoms = torch.tensor(num_atoms_list)
    batch = torch.tensor(batch_idx)

    # This should trigger the memory guard and fall back to sequential
    with torch.no_grad():
        result, logits = decoder._predict_distance_matrices(x, batch, num_atoms)

    # Should still produce correct output (via fallback)
    assert len(result) == num_graphs, (
        f"Expected {num_graphs} matrices, got {len(result)}"
    )
    for i, D in enumerate(result):
        assert D.shape == (20, 20), f"Graph {i} has wrong shape: {D.shape}"
        assert torch.allclose(D, D.T, atol=1e-6), f"Graph {i} not symmetric"
        assert (D >= 0).all(), f"Graph {i} has negative distances"
        assert torch.allclose(torch.diag(D), torch.zeros(20), atol=1e-6), (
            f"Graph {i} diagonal not zero"
        )


@pytest.mark.unit
def test_vectorized_flag_controls_behavior():
    """Test that use_vectorized flag correctly controls behavior."""
    torch.manual_seed(42)

    # Small test case
    num_atoms_list = [5, 7, 10]
    total_atoms = sum(num_atoms_list)
    batch_idx = []
    token_idx = []
    for i, n in enumerate(num_atoms_list):
        batch_idx.extend([i] * n)
        token_idx.extend(list(range(n)))

    encoded_batch = {
        "x": torch.randn(total_atoms, 128),
        "num_atoms": torch.tensor(num_atoms_list),
        "batch": torch.tensor(batch_idx),
        "token_idx": torch.tensor(token_idx),
    }

    # Test with use_vectorized=True
    decoder_vec = DistanceMatrixDecoder(d_model=128, num_layers=2, use_vectorized=True)
    decoder_vec.eval()
    with torch.no_grad():
        output_vec = decoder_vec(encoded_batch)

    # Test with use_vectorized=False
    # Need to use same weights for fair comparison
    decoder_seq = DistanceMatrixDecoder(d_model=128, num_layers=2, use_vectorized=False)
    decoder_seq.load_state_dict(decoder_vec.state_dict())
    decoder_seq.eval()
    with torch.no_grad():
        output_seq = decoder_seq(encoded_batch)

    # Outputs should be identical
    assert len(output_vec["distance_matrix"]) == len(output_seq["distance_matrix"])
    for D_vec, D_seq in zip(
        output_vec["distance_matrix"], output_seq["distance_matrix"], strict=False
    ):
        assert torch.allclose(D_vec, D_seq, atol=1e-5, rtol=1e-4), (
            f"Vectorized and sequential outputs differ: max_diff={torch.abs(D_vec - D_seq).max()}"
        )


@pytest.mark.unit
def test_vectorized_gradient_flow():
    """Test that gradients backpropagate correctly through vectorized path."""
    decoder = DistanceMatrixDecoder(d_model=128, num_layers=2, use_vectorized=True)
    decoder.train()

    # Small batch
    num_atoms_list = [3, 4, 5]
    total_atoms = sum(num_atoms_list)
    batch_idx = []
    token_idx = []
    for i, n in enumerate(num_atoms_list):
        batch_idx.extend([i] * n)
        token_idx.extend(list(range(n)))

    encoded_batch = {
        "x": torch.randn(total_atoms, 128, requires_grad=True),
        "num_atoms": torch.tensor(num_atoms_list),
        "batch": torch.tensor(batch_idx),
        "token_idx": torch.tensor(token_idx),
    }

    # Forward pass
    output = decoder(encoded_batch)

    # Compute dummy loss (sum of all distances)
    loss = sum(D.sum() for D in output["distance_matrix"])

    # Backward pass
    loss.backward()

    # Check that gradients exist and are non-zero
    assert encoded_batch["x"].grad is not None, "No gradient for input x"
    assert encoded_batch["x"].grad.abs().sum() > 0, "Gradient is zero"

    # Check that MLP parameters have gradients
    for name, param in decoder.distance_mlp.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Gradient is zero for {name}"


@pytest.mark.unit
def test_classifier_output_shape():
    """Test classifier outputs correct shapes and exists in output."""
    decoder = DistanceMatrixDecoder(
        d_model=256, num_layers=2, classifier_hidden_dims=[256, 128]
    )

    encoded_batch = {
        "x": torch.randn(10, 256),
        "num_atoms": torch.tensor([4, 6]),
        "batch": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        "token_idx": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5]),
    }

    output = decoder(encoded_batch)

    assert "distance_classifier_logits" in output
    assert len(output["distance_classifier_logits"]) == 2
    assert output["distance_classifier_logits"][0].shape == (4, 4)
    assert output["distance_classifier_logits"][1].shape == (6, 6)


@pytest.mark.unit
def test_classifier_symmetry():
    """Test classifier logits are symmetric (same as distances)."""
    decoder = DistanceMatrixDecoder(d_model=256, num_layers=2)

    encoded_batch = {
        "x": torch.randn(5, 256),
        "num_atoms": torch.tensor([5]),
        "batch": torch.zeros(5, dtype=torch.long),
        "token_idx": torch.arange(5),
    }

    output = decoder(encoded_batch)
    logits = output["distance_classifier_logits"][0]

    assert torch.allclose(logits, logits.T, atol=1e-6)


@pytest.mark.unit
def test_classifier_probability_range():
    """Test sigmoid(logits) produces valid probabilities in [0,1]."""
    decoder = DistanceMatrixDecoder(d_model=256, num_layers=2)

    encoded_batch = {
        "x": torch.randn(5, 256),
        "num_atoms": torch.tensor([5]),
        "batch": torch.zeros(5, dtype=torch.long),
        "token_idx": torch.arange(5),
    }

    output = decoder(encoded_batch)
    logits = output["distance_classifier_logits"][0]
    probs = torch.sigmoid(logits)

    assert (probs >= 0).all() and (probs <= 1).all()


@pytest.mark.unit
def test_sparse_loss_masking():
    """Test that distance loss is only computed on pairs < threshold."""
    from src.vae_module.vae_module import VAEModule

    # Synthetic data: distances [5, 15] with threshold=10
    pred_matrices = [
        torch.tensor([[0.0, 5.0, 15.0], [5.0, 0.0, 20.0], [15.0, 20.0, 0.0]])
    ]
    true_matrices = [
        torch.tensor([[0.0, 5.0, 15.0], [5.0, 0.0, 20.0], [15.0, 20.0, 0.0]])
    ]
    classifier_logits = [torch.zeros(3, 3)]

    # Mock VAEModule instance (just for calling the method)
    class MockVAE:
        pass

    vae = MockVAE()

    losses = VAEModule._compute_distance_matrix_loss(
        vae, pred_matrices, true_matrices, classifier_logits, distance_threshold=10.0
    )

    # Only pair (0,1) with distance=5 should contribute to regression loss
    # Since pred=true for that pair, distance_loss should be ~0
    assert torch.allclose(losses["distance_loss"], torch.tensor(0.0), atol=1e-5)

    # Classifier loss should be non-zero (even with perfect distances, logits are random)
    assert losses["classifier_loss"] > 0


@pytest.mark.unit
def test_gradient_flow_through_classifier():
    """Test gradients flow correctly through classifier MLP."""
    decoder = DistanceMatrixDecoder(d_model=128, num_layers=2)
    decoder.train()

    encoded_batch = {
        "x": torch.randn(5, 128, requires_grad=True),
        "num_atoms": torch.tensor([5]),
        "batch": torch.zeros(5, dtype=torch.long),
        "token_idx": torch.arange(5),
    }

    output = decoder(encoded_batch)
    loss = output["distance_classifier_logits"][0].sum()
    loss.backward()

    # Check input gradients exist
    assert encoded_batch["x"].grad is not None
    assert encoded_batch["x"].grad.abs().sum() > 0

    # Check classifier MLP has gradients
    for param in decoder.classifier_mlp.parameters():
        assert param.grad is not None


@pytest.mark.unit
def test_vectorized_vs_sequential_equivalence_with_classifier():
    """Verify vectorized produces identical classifier results to sequential."""
    torch.manual_seed(42)  # For reproducibility

    # Test case
    num_atoms_list = [4, 8, 12]
    total_atoms = sum(num_atoms_list)
    batch_idx = []
    token_idx = []
    for i, n in enumerate(num_atoms_list):
        batch_idx.extend([i] * n)
        token_idx.extend(list(range(n)))

    x = torch.randn(total_atoms, 256)
    num_atoms = torch.tensor(num_atoms_list)
    batch = torch.tensor(batch_idx)

    # Create decoder (use same instance for fair comparison)
    decoder = DistanceMatrixDecoder(
        d_model=256,
        num_layers=2,
        use_vectorized=False,  # Will manually test both paths
    )
    decoder.eval()  # Disable dropout

    # Compare outputs
    with torch.no_grad():
        seq_distances, seq_logits = decoder._predict_distance_matrices_sequential(
            x, batch, num_atoms
        )
        vec_distances, vec_logits = decoder._predict_distance_matrices_vectorized(
            x, batch, num_atoms
        )

    # Check distances match
    assert len(seq_distances) == len(vec_distances)
    for i, (D_seq, D_vec) in enumerate(zip(seq_distances, vec_distances, strict=False)):
        assert torch.allclose(D_seq, D_vec, atol=1e-5, rtol=1e-4), (
            f"Distance mismatch in graph {i}"
        )

    # Check classifier logits match
    assert len(seq_logits) == len(vec_logits)
    for i, (L_seq, L_vec) in enumerate(zip(seq_logits, vec_logits, strict=False)):
        assert torch.allclose(L_seq, L_vec, atol=1e-5, rtol=1e-4), (
            f"Logits mismatch in graph {i}: max_diff={torch.abs(L_seq - L_vec).max():.6f}"
        )
