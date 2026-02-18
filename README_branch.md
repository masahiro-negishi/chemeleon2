# Branch: feat/mace-encoder

## Overview

This branch introduces a distance matrix-based decoder for the VAE module, along with MACE encoder integration.

---

## New Features

### Distance Matrix Decoder

Alternative decoder that predicts pairwise distance matrices instead of fractional coordinates. Better aligned with MACE embeddings which encode relative geometry.

**Key Features**:
- Permutation-invariant pairwise features: `[h_i + h_j || h_i * h_j]`
- Symmetric, non-negative distance matrices with zero diagonal
- Sparse loss with binary classifier (close/far pairs)
- Vectorized computation with automatic fallback for large batches

### MACE Encoder Integration

Integration with MACE (Machine learning Atomic Cluster Expansion) pre-computed features for improved geometric understanding.

---

## Files Added

| File | Lines | Purpose |
|------|-------|---------|
| `src/vae_module/utils.py` | 28 | Shared VAE utilities (positional embeddings) |
| `src/vae_module/decoders/distance_matrix.py` | 436 | Distance matrix decoder implementation |
| `src/utils/distance_matrix.py` | 44 | Distance matrix computation utilities |
| `configs/vae_module/decoder/distance_matrix.yaml` | 16 | Distance decoder configuration |
| `configs/experiment/mp_20/vae_mace_distance.yaml` | 37 | MACE+distance experiment config |
| `tests/baseline/test_vae_distance_matrix.py` | 288 | Baseline validation tests |
| `tests/unit/test_distance_matrix_decoder.py` | 607 | Comprehensive decoder unit tests |
| `tests/unit/test_distance_matrix_utils.py` | 149 | Utility function tests |

**Total**: 8 new files, ~1,600 lines of new code (including comprehensive tests)

---

## Files Modified

| File | Changes |
|------|---------|
| `src/vae_module/vae_module.py` | Added distance matrix loss computation, removed dead backward compat code |
| `src/vae_module/decoders/transformer.py` | Now imports shared `get_index_embedding` utility |
| `configs/vae_module/vae_module.yaml` | Added distance_threshold, distance_regression, distance_classifier weights |
| `tests/conftest.py` | Added shared fixtures: `simple_encoder`, `simple_distance_decoder`, `vae_distance_model` |

---

## Architecture Details

### Distance Matrix Decoder Architecture

```
Latent vectors (num_nodes, latent_dim)
    ↓
Transformer layers (num_nodes, d_model)
    ↓
Pairwise features: [h_i + h_j || h_i * h_j]
    ↓
┌─────────────────────┬─────────────────────┐
│  Distance MLP       │  Classifier MLP     │
│  (pairwise, 2d)     │  (pairwise, 2d)     │
│  → scalar distance  │  → logits (close?)  │
└─────────────────────┴─────────────────────┘
    ↓                       ↓
Distance matrix        Binary classification
(N, N) symmetric       (N, N) logits
```

### Loss Components

1. **Distance Regression Loss**: MSE on pairs below threshold
   - Focuses on close pairs (< 5Å by default)
   - Weighted by `distance_regression` factor

2. **Distance Classifier Loss**: BCE on all pairs
   - Binary classification: is pair close (<threshold)?
   - Weighted by `distance_classifier` factor
   - Helps model learn geometric structure

3. **Reconstruction Losses**: atom types, lattice parameters
   - Standard VAE reconstruction terms
   - Weighted separately in config

---

## MACE Encoder Configuration

Example configuration for MACE encoder with distance decoder:

```yaml
# configs/experiment/mp_20/vae_mace_distance.yaml
defaults:
  - override /vae_module/encoder: precomputed_mace
  - override /vae_module/decoder: distance_matrix

data:
  mace_embeddings: true
  mace_precompute_params:
    model: "mh-1"            # MACE model name
    head: "omat_pbe"         # Multihead model head
    hidden_dim: 512          # Expected embedding dimension
    device: "cuda"           # Computation device

vae_module:
  loss_weights:
    atom_types: 1.0
    lengths: 1.0
    angles: 10.0
    frac_coords: 0.0         # Disabled (using distance)
    distance_regression: 1.0
    distance_classifier: 1.0
    kl: 0.0                  # Disabled for precomputed MACE
    fa: 0.0                  # Redundant with distance loss
```
