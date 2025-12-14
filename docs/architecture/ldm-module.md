# LDM Module

The Latent Diffusion Model module ([`src/ldm_module/`](https://github.com/hspark1212/chemeleon2/tree/main/src/ldm_module)) learns to generate crystal structures by denoising in the VAE's latent space.

## Architecture

```{mermaid}
flowchart LR
    subgraph Training
        direction TB
        A[Latent z from VAE]
        B[Add Noise at timestep t]
        C[DiT Denoiser]
        D[Predict Noise]
        E[Diffusion Loss]
        A --> B --> C --> D --> E
    end

    subgraph Sampling
        direction TB
        F[Random Noise]
        G[Iterative Denoising]
        H[Clean Latent z]
        I[VAE Decoder]
        J[Crystal Structure]
        F --> G --> H --> I --> J
    end

    Training ~~~ Sampling

    style H fill:#ffffcc
```

## Key Classes

### LDMModule

PyTorch Lightning module for the latent diffusion model ([`src/ldm_module/ldm_module.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/ldm_module/ldm_module.py)):

```python
from src.ldm_module import LDMModule

# Load pre-trained LDM
ldm = LDMModule.load_from_checkpoint("path/to/checkpoint.ckpt", weights_only=False)

# Sample new structures
batch_gen = ldm.sample(batch, sampling_steps=50)
```

**Key Methods:**
- `calculate_loss(batch, training=True)` - Computes diffusion training loss
- `sample(batch, sampler="ddim", sampling_steps=50, cfg_scale=2.0, ...)` - Generates structures via DDPM or DDIM sampling

### DiT (Diffusion Transformer)

The denoiser architecture based on Meta's DiT:

- Transformer blocks with adaptive layer norm (adaLN)
- Timestep and condition embeddings
- Support for variable-length sequences with masking

```python
# DiT configuration
denoiser:
  _target_: src.ldm_module.denoisers.DiT
  hidden_size: 512
  depth: 12
  num_heads: 8
  mlp_ratio: 4.0
```

### Gaussian Diffusion

Implements the diffusion process:

- Forward process: gradually adds noise
- Reverse process: learned denoising
- Supports DDPM and DDIM sampling

## Conditional Generation

The LDM supports conditioning on:

| Condition Type | Description | Config |
|----------------|-------------|--------|
| Composition | Chemical formula guidance | `ldm_composition` |
| Band gap | Property-conditioned generation | `ldm_bandgap` |
| Custom | Extensible condition module | Custom config |

### Classifier-Free Guidance (CFG)

```bash
# Sample with CFG
python src/sample.py \
    ldm_ckpt=path/to/ldm.ckpt \
    cfg_scale=2.0
```

## Configuration

See [`configs/ldm_module/`](https://github.com/hspark1212/chemeleon2/tree/main/configs/ldm_module) for LDM configurations:

```yaml
# configs/ldm_module/ldm_module.yaml (default)
_target_: src.ldm_module.ldm_module.LDMModule
denoiser:
  _target_: src.ldm_module.denoisers.dit.DiT
  hidden_size: 768
  depth: 12
  num_heads: 12
diffusion_configs:
  diffusion_steps: 1000
  learn_sigma: true
```

## Training

```bash
# Unconditional LDM
python src/train_ldm.py experiment=mp_20/ldm_null

# Composition-conditioned LDM
python src/train_ldm.py experiment=mp_20/ldm_composition
```

Training script: [`src/train_ldm.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_ldm.py)

See [Training Guide](../user-guide/training/index.md) for more details.
