# LDM Training

The Latent Diffusion Model (LDM) is the second stage of the Chemeleon2 pipeline. It learns to generate crystal structures by denoising in the VAE's latent space.

## What LDM Does

The LDM is the second stage of Chemeleon2 that learns to generate crystal structures by denoising in the VAE's latent space. For architectural details, see [LDM Module](../../architecture/ldm-module.md).

Key components (see [`src/ldm_module/ldm_module.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/ldm_module/ldm_module.py)):
- **Diffusion Transformer (DiT)**: Predicts noise at each timestep
- **DDPM/DDIM Sampling**: Iteratively denoises random noise
- **Conditioning**: Optional guidance from composition or properties

## Prerequisites

LDM training requires a **trained VAE checkpoint**. The VAE encodes crystal structures into the latent space where the LDM operates.

```yaml
# In config files
ldm_module:
  vae_ckpt_path: ${hub:mp_20_vae}  # Or use local path
```

```bash
# In CLI
python src/train_ldm.py ldm_module.vae_ckpt_path='${hub:mp_20_vae}'
```

See [Checkpoint Management](index.md#checkpoint-management) for available checkpoints.

## Quick Start

```bash
# Train unconditional LDM (src/train_ldm.py)
python src/train_ldm.py experiment=mp_20/ldm_null
```

Training script: [`src/train_ldm.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/train_ldm.py)
Example config: [`configs/experiment/mp_20/ldm_null.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/experiment/mp_20/ldm_null.yaml)

:::{tip}
**Example training run**: See [this W&B run](https://wandb.ai/hspark1212/chemeleon2/groups/train_ldm%2Fmp_20/runs/4tfw67aq) for a successful LDM training example on MP-20 dataset.
:::

## Training Modes

:::{note}
LDM supports conditional generation using **classifier-free guidance**, which guides the diffusion process during training. This is different from RL-based optimization (the next stage). But RL fine-tuning is more recommended for conditioned generation.
:::

### Unconditional Generation

Generate diverse structures without any guidance:

```bash
python src/train_ldm.py experiment=mp_20/ldm_null
```

### Composition-Conditioned Generation

Guide generation with target chemical composition:

```bash
python src/train_ldm.py experiment=mp_20/ldm_composition
```

### Property-Conditioned Generation

Guide generation with target property values (e.g., band gap):

```bash
python src/train_ldm.py experiment=alex_mp_20_bandgap/ldm_bandgap
```

## Training Commands

### Basic Training

```bash
# Use experiment config
python src/train_ldm.py experiment=mp_20/ldm_null

# Override checkpoint path
python src/train_ldm.py experiment=mp_20/ldm_null \
    ldm_module.vae_ckpt_path=ckpts/my_vae.ckpt

# Override training parameters
python src/train_ldm.py experiment=mp_20/ldm_null \
    trainer.max_epochs=500 \
    data.batch_size=64
```

### Advanced: LoRA Fine-tuning

Fine-tune a pre-trained LDM with Low-Rank Adaptation (LoRA):

```bash
python src/train_ldm.py experiment=alex_mp_20_bandgap/ldm_bandgap_lora
```

LoRA enables efficient fine-tuning by only updating low-rank adapter weights instead of all model parameters. This approach:
- **Reduces memory usage**: Only adapter weights require gradients
- **Faster training**: Fewer parameters to update
- **Prevents catastrophic forgetting**: Base model weights remain frozen

Use LoRA when fine-tuning a pre-trained LDM on new datasets or conditions.

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_diffusion_steps` | 1000 | Number of diffusion timesteps |
| `hidden_dim` | 768 | DiT hidden dimension (dit_b config) |
| `num_layers` | 12 | Number of DiT layers (depth) |
| `num_heads` | 12 | Number of attention heads |

### Example Config Override

```bash
python src/train_ldm.py experiment=mp_20/ldm_null \
    ldm_module.num_diffusion_steps=500 \
    ldm_module.hidden_dim=768
```

## Available Experiments

| Experiment | Dataset | Condition | Description |
|------------|---------|-----------|-------------|
| `mp_20/ldm_null` | MP-20 | None | Unconditional generation |
| `mp_20/ldm_composition` | MP-20 | Composition | Composition-guided |
| `alex_mp_20_bandgap/ldm_bandgap` | Alex MP-20 | Band gap | Property-guided |
| `alex_mp_20_bandgap/ldm_bandgap_lora` | Alex MP-20 | Band gap | LoRA fine-tuning |

## Training Tips

### Monitoring

Key metrics to watch in WandB:
- `train/loss`: Diffusion loss (should decrease)
- `val/loss`: Validation loss (check for overfitting)

### Typical Training

- **Duration**: Up to 5000 epochs (default), with early stopping after 200 epochs without improvement
- **Batch size**: 256 (default), can be reduced to 32-128 for limited GPU memory
- **Learning rate**: 1e-4 (default)

## Next Steps

After training LDM:
1. Note the checkpoint path
2. Option A: Proceed to [RL Training](rl.md) to fine-tune with rewards
3. Option B: Use directly for generation (see [Evaluation](../evaluation.md))
