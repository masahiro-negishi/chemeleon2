# VAE Module

The Variational Autoencoder module ([`src/vae_module/`](https://github.com/hspark1212/chemeleon2/tree/main/src/vae_module)) encodes crystal structures into continuous latent representations and decodes them back.

## Architecture

```{mermaid}
flowchart TB
    A[Crystal Structure<br/>atom_types, frac_coords, lattice]

    B[Atom Type Embedding]
    C[Transformer Encoder]
    D[Quant Conv → μ, σ]

    E[Latent z ~ N μ, σ²]

    F[Post Quant Conv]
    G[Transformer Decoder]
    H[Reconstructed Structure<br/>atom_types, frac_coords, lengths, angles]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H

    style E fill:#ffffcc
```

## Key Classes

### VAEModule

The main PyTorch Lightning module implementing the VAE ([`src/vae_module/vae_module.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/vae_module/vae_module.py)):

```python
from src.vae_module import VAEModule

# Load pre-trained VAE
vae = VAEModule.load_from_checkpoint("path/to/checkpoint.ckpt", weights_only=False)

# Encode crystal batch to latent distribution
encoded = vae.encode(batch)
posterior = encoded["posterior"]
z = posterior.sample()

# Decode latent vectors to crystal properties
encoded["x"] = z
decoder_out = vae.decode(encoded)

# Reconstruct crystal structures
batch_recon = vae.reconstruct(decoder_out, batch)
```

**Key Methods:**
- `encode(batch)` - Encodes crystal batch to latent distribution (returns dict with "posterior")
- `decode(encoded)` - Decodes latent dict to crystal properties (returns dict)
- `sample(batch, return_atoms=False, return_structures=False)` - Generates random samples from the latent space
- `reconstruct(decoder_out, batch)` - Reconstructs CrystalBatch from decoder output

### DiagonalGaussianDistribution

Represents the latent distribution with diagonal covariance:

- Parameterized by mean and log-variance
- Supports sampling and KL divergence computation

## Loss Functions

The VAE training minimizes:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{recon} + \lambda_2 \mathcal{L}_{KL}$$

Where:
- $\mathcal{L}_{recon}$: Reconstruction loss (atom types, coordinates, lattice)
- $\mathcal{L}_{KL}$: KL divergence regularization

## Configuration

See [`configs/vae_module/`](https://github.com/hspark1212/chemeleon2/tree/main/configs/vae_module) for VAE configurations:

```yaml
# configs/vae_module/vae_module.yaml (default)
_target_: src.vae_module.vae_module.VAEModule
encoder:
  _target_: src.vae_module.encoders.transformer.TransformerEncoder
  d_model: 512
  nhead: 8
  num_layers: 8
decoder:
  _target_: src.vae_module.decoders.transformer.TransformerDecoder
  d_model: 512
  nhead: 8
  num_layers: 8
latent_dim: 8
```

## Training

```bash
python src/train_vae.py experiment=mp_20/vae_dng
```

Training script: [`src/train_vae.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_vae.py)

See [Training Guide](../user-guide/training/index.md) for more details.
