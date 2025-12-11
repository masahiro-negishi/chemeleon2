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
vae = VAEModule.load_from_checkpoint("path/to/checkpoint.ckpt")

# Encode crystal batch to latent distribution
posterior = vae.encode(batch)
z = posterior.sample()

# Decode latent vectors to crystal properties
decoder_out = vae.decode(z, batch)
```

**Key Methods:**
- `encode(batch)` - Encodes crystal batch to latent distribution
- `decode(encoded, batch)` - Decodes latent vectors to crystal properties
- `sample(batch)` - Generates random samples from the latent space
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
# configs/vae_module/vae_dng.yaml
_target_: src.vae_module.VAEModule
encoder:
  _target_: src.vae_module.encoders.TransformerEncoder
  d_model: 256
  nhead: 8
  num_layers: 6
decoder:
  _target_: src.vae_module.decoders.TransformerDecoder
  d_model: 256
  nhead: 8
  num_layers: 6
latent_dim: 256
```

## Training

```bash
python src/train_vae.py experiment=mp_20/vae_dng
```

Training script: [`src/train_vae.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_vae.py)

See [Training Guide](../user-guide/training/index.md) for more details.
