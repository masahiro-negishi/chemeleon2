"""Baseline tests for VAE module with distance matrix decoder.

This test file validates the VAE module with distance matrix decoder
to ensure the distance-based reconstruction works correctly.
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pymatgen.analysis.structure_matcher import StructureMatcher

from src.vae_module.decoders.distance_matrix import DistanceMatrixDecoder
from src.vae_module.vae_module import VAEModule


@pytest.mark.smoke
@pytest.mark.baseline
def test_vae_distance_instantiation(vae_distance_model, device) -> None:
    """Test VAE model with distance decoder instantiation."""
    assert isinstance(vae_distance_model, VAEModule)
    assert isinstance(vae_distance_model.encoder, nn.Module)
    assert isinstance(vae_distance_model.decoder, DistanceMatrixDecoder)
    assert vae_distance_model.quant_conv is not None
    assert vae_distance_model.post_quant_conv is not None
    assert str(vae_distance_model.device).startswith(device)


@pytest.mark.smoke
@pytest.mark.baseline
def test_vae_distance_forward_pass_shapes(
    vae_distance_model, dummy_crystal_batch, device
) -> None:
    """Test VAE forward pass with distance decoder."""
    # Create small batch for smoke test
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")
    batch = batch.to(device)

    # Forward pass
    vae_distance_model.eval()
    with torch.no_grad():
        decoder_out, encoded = vae_distance_model(batch)

    # Validate encoded outputs
    assert "posterior" in encoded
    assert "z" in encoded
    assert encoded["z"].shape[0] == batch.num_nodes
    assert encoded["z"].shape[1] == vae_distance_model.hparams.latent_dim

    # Validate decoder outputs
    assert "atom_types" in decoder_out
    assert "lengths" in decoder_out
    assert "angles" in decoder_out
    assert "distance_matrix" in decoder_out
    assert "distance_classifier_logits" in decoder_out
    assert "frac_coords" in decoder_out

    # Shape validation
    assert decoder_out["atom_types"].shape[0] == batch.num_nodes
    assert decoder_out["lengths"].shape[0] == batch.num_graphs
    assert decoder_out["angles"].shape[0] == batch.num_graphs
    assert len(decoder_out["distance_matrix"]) == batch.num_graphs

    # Check distance matrices are lists of tensors
    for i, dist_matrix in enumerate(decoder_out["distance_matrix"]):
        n_atoms = batch.num_atoms[i].item()
        assert dist_matrix.shape == (n_atoms, n_atoms)

    # Check classifier logits are lists of tensors with correct shapes
    assert len(decoder_out["distance_classifier_logits"]) == batch.num_graphs
    for i, logits in enumerate(decoder_out["distance_classifier_logits"]):
        n_atoms = batch.num_atoms[i].item()
        assert logits.shape == (n_atoms, n_atoms)

    # Check frac_coords are dummy zeros
    assert torch.allclose(
        decoder_out["frac_coords"], torch.zeros_like(decoder_out["frac_coords"])
    )


@pytest.mark.smoke
@pytest.mark.baseline
def test_vae_distance_loss_calculation(
    vae_distance_model, dummy_crystal_batch, device
) -> None:
    """Test VAE loss calculation with distance matrix loss."""
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")
    batch = batch.to(device)

    vae_distance_model.eval()
    with torch.no_grad():
        loss_dict = vae_distance_model.calculate_loss(batch, training=False)

    # Validate loss dictionary
    assert "total_loss" in loss_dict
    assert torch.isfinite(loss_dict["total_loss"])
    assert loss_dict["total_loss"] > 0

    # Validate loss components
    expected_keys = [
        "loss_atom_types",
        "loss_lengths",
        "loss_angles",
        "loss_distance_matrix",
        "loss_distance_classifier",
        "loss_distance_regression",
        "loss_kl",
    ]
    for key in expected_keys:
        assert key in loss_dict, f"Missing key: {key}"
        assert torch.isfinite(loss_dict[key]), f"Non-finite value for {key}"

    # Check that distance_matrix loss is non-zero
    assert loss_dict["loss_distance_matrix"] > 0

    # Check that classifier and regression losses are non-zero
    assert loss_dict["loss_distance_classifier"] > 0
    assert loss_dict["loss_distance_regression"] > 0

    # Check that frac_coords loss is zero (disabled)
    assert loss_dict["loss_frac_coords"] == 0


@pytest.mark.baseline
def test_vae_distance_matrix_constraints(
    vae_distance_model, dummy_crystal_batch, device
) -> None:
    """Test that predicted distance matrices satisfy constraints."""
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")
    batch = batch.to(device)

    vae_distance_model.eval()
    with torch.no_grad():
        decoder_out, _ = vae_distance_model(batch)

    distance_matrices = decoder_out["distance_matrix"]

    for D in distance_matrices:
        # Check symmetry
        assert torch.allclose(D, D.T, atol=1e-5), "Distance matrix not symmetric"

        # Check non-negativity
        assert (D >= 0).all(), "Distance matrix has negative values"

        # Check zero diagonal
        assert torch.allclose(
            torch.diag(D), torch.zeros(D.shape[0], device=device), atol=1e-5
        ), "Distance matrix diagonal not zero"


@pytest.mark.baseline
@pytest.mark.slow
def test_vae_distance_overfit_single_batch(
    vae_distance_model, dummy_crystal_batch, seed_everything, device
) -> None:
    """Test VAE with distance decoder can overfit on a single batch.

    Critical validation test: verifies that the distance matrix decoder
    can learn to reconstruct distance matrices from a single batch.
    """
    seed_everything(42)

    # Create single batch
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")
    batch = batch.to(device)

    # Set model to training mode
    vae_distance_model.train()

    # Configure optimizer
    optimizer = torch.optim.Adam(vae_distance_model.parameters(), lr=1e-3)

    # Record initial loss
    with torch.no_grad():
        initial_loss_dict = vae_distance_model.calculate_loss(batch, training=False)
        initial_loss = initial_loss_dict["total_loss"].item()
        initial_dist_loss = initial_loss_dict["loss_distance_matrix"].item()

    # Train for 200 iterations
    num_iterations = 200
    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss_dict = vae_distance_model.calculate_loss(batch, training=True)
        loss = loss_dict["total_loss"]
        loss.backward()
        optimizer.step()

    # Record final loss
    vae_distance_model.eval()
    with torch.no_grad():
        final_loss_dict = vae_distance_model.calculate_loss(batch, training=False)
        final_loss = final_loss_dict["total_loss"].item()
        final_dist_loss = final_loss_dict["loss_distance_matrix"].item()

    # Verify loss decreased significantly
    assert final_loss < initial_loss * 0.35, (
        f"Failed to overfit single batch: "
        f"initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"
    )

    # Verify distance matrix loss decreased
    assert final_dist_loss < initial_dist_loss * 0.35, (
        f"Distance matrix loss did not decrease enough: "
        f"initial={initial_dist_loss:.4f}, final={final_dist_loss:.4f}"
    )


@pytest.mark.baseline
def test_vae_distance_structure_matching_skipped(
    vae_distance_model, dummy_crystal_batch, device
) -> None:
    """Test that structure matching returns NaN for distance decoder."""
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")
    batch = batch.to(device)

    vae_distance_model.eval()
    with torch.no_grad():
        structure_matching = vae_distance_model._compute_structure_matching(batch)

    # Should return NaN since distance decoder doesn't produce valid coordinates
    import math

    assert math.isnan(structure_matching), (
        "Structure matching should return NaN for distance matrix decoder"
    )


@pytest.mark.baseline
def test_vae_distance_separate_weights(
    simple_encoder, simple_distance_decoder, dummy_crystal_batch, device
) -> None:
    """Test VAE with separate weights for regression and classifier."""
    # Create model with separate weights (regression=2.0, classifier=0.5)
    model = VAEModule(
        encoder=simple_encoder,
        decoder=simple_distance_decoder,
        latent_dim=64,
        distance_threshold=10.0,
        loss_weights=OmegaConf.create(  # type: ignore
            {
                "atom_types": 0.0,  # Disabled to isolate distance losses
                "lengths": 0.0,
                "angles": 0.0,
                "frac_coords": 0.0,
                "distance_regression": 2.0,  # 2x weight
                "distance_classifier": 0.5,  # 0.5x weight
                "kl": 0.0,
                "fa": 0.0,
            }
        ),
        augmentation=OmegaConf.create(  # type: ignore
            {
                "translate": False,
                "rotate": False,
            }
        ),
        noise=OmegaConf.create(  # type: ignore
            {
                "ratio": 0.0,
                "corruption_scale": 0.0,
            }
        ),
        atom_type_predict=True,
        structure_matcher=StructureMatcher(),
        optimizer=torch.optim.Adam,  # type: ignore
        scheduler=None,  # type: ignore
    ).to(device)

    # Create batch and compute loss
    batch = dummy_crystal_batch(batch_size=2, num_atom_distribution="mp-20")
    batch = batch.to(device)

    model.eval()
    with torch.no_grad():
        loss_dict = model.calculate_loss(batch, training=False)

    # Extract component losses
    loss_regression = loss_dict["loss_distance_regression"].item()
    loss_classifier = loss_dict["loss_distance_classifier"].item()
    total_loss = loss_dict["total_loss"].item()

    # Verify total loss reflects separate weights
    # Total = 2.0 * regression + 0.5 * classifier (other losses disabled)
    expected_total = 2.0 * loss_regression + 0.5 * loss_classifier
    assert abs(total_loss - expected_total) < 1e-5, (
        f"Total loss {total_loss:.6f} doesn't match expected "
        f"{expected_total:.6f} (2.0*{loss_regression:.6f} + 0.5*{loss_classifier:.6f})"
    )
