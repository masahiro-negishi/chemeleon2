# Training Overview

This guide covers how to train Chemeleon2 models. The framework implements a three-stage training pipeline where each stage builds upon the previous one.

## Training Pipeline

```{mermaid}
flowchart LR
    subgraph Pipeline["Three-Stage Training"]
        A[VAE Training] --> B[LDM Training]
        B --> C[RL Fine-tuning]
    end
```

| Stage | Purpose |
|-------|---------|
| **VAE** | Encode crystal structures into latent space |
| **LDM** | Learn to generate in latent space via diffusion |
| **RL** | Fine-tune LDM with reward functions using RL algorithm|
| **Predictor** | Predict properties from latent vectors (optional) with VAE encoder |

## Configuration System

Chemeleon2 uses [Hydra](https://hydra.cc/) for configuration management. All configs are in the [`configs/`](https://github.com/hspark1212/chemeleon2/tree/main/configs) directory.

### Directory Structure

```
configs/
├── train_vae.yaml                              # configs/train_vae.yaml
├── train_ldm.yaml                              # configs/train_ldm.yaml
├── train_rl.yaml                               # configs/train_rl.yaml
├── train_predictor.yaml                        # configs/train_predictor.yaml
├── experiment/                                 # Experiment-specific overrides
│   ├── mp_20/
│   │   ├── vae_dng.yaml                        # configs/experiment/mp_20/vae_dng.yaml
│   │   ├── ldm_null.yaml                       # configs/experiment/mp_20/ldm_null.yaml
│   │   └── rl_dng.yaml                         # configs/experiment/mp_20/rl_dng.yaml
│   └── alex_mp_20_bandgap/
│       ├── predictor_dft_band_gap.yaml         # configs/experiment/alex_mp_20_bandgap/predictor_dft_band_gap.yaml
│       └── rl_bandgap.yaml                     # configs/experiment/alex_mp_20_bandgap/rl_bandgap.yaml
├── data/                                       # Dataset configurations
├── vae_module/                                 # VAE architecture configs
├── ldm_module/                                 # LDM architecture configs
├── rl_module/                                  # RL configs
├── trainer/                                    # PyTorch Lightning trainer
├── logger/                                     # WandB logging (configs/logger/wandb.yaml)
└── callbacks/                                  # Training callbacks
```

### View Resolved Configuration

Check the fully resolved config without running training:

```bash
python src/train_ldm.py experiment=mp_20/ldm_null --cfg job
```

:::{tip} How Experiment Configs Work
The above example runs [`src/train_ldm.py`](https://github.com/hspark1212/chemeleon2/tree/main/src/train_ldm.py), so Hydra loads configurations in this order:

1. **Base config loaded first**: [`configs/train_ldm.yaml`](https://github.com/hspark1212/chemeleon2/tree/main/configs/train_ldm.yaml)
   - Specified in the `@hydra.main(config_name="train_ldm.yaml")` decorator in [`src/train_ldm.py`](https://github.com/hspark1212/chemeleon2/blob/6e869607fc9ccbbd34526ec42f745265b8851a84/src/train_ldm.py#L13)
   - Defines default settings for all components (data, model, callbacks, trainer, etc.)

2. **Experiment config applied second**: [`configs/experiment/mp_20/ldm_null.yaml`](https://github.com/hspark1212/chemeleon2/tree/main/configs/experiment/mp_20/ldm_null.yaml)
   - Loaded via `experiment=mp_20/ldm_null` argument
   - Overrides specific parameters:
     - `ldm_module.vae_ckpt_path: ${hub:mp_20_vae}` - downloads pre-trained VAE from HuggingFace
     - `data.target_condition: null` - trains unconditional LDM (no property constraints)
     - `logger.wandb.tags: ["ldm", "dng", "null_condition"]` - adds experiment tracking tags

3. **Final config**: Base config + Experiment overrides (only specified values are replaced)

This hierarchical approach lets you maintain clean experiment configurations without duplicating the entire config file. You can add any parameter from the base config to your experiment config to override its default value.
:::

### Override Syntax

Override any config parameter from the command line:

```bash
# Override single parameter
python src/train_vae.py trainer.max_epochs=100

# Override multiple parameters
python src/train_ldm.py data.batch_size=64 trainer.max_epochs=500

# Use experiment config (loads all overrides from file)
python src/train_vae.py experiment=mp_20/vae_dng
```

## Checkpoint Management

Chemeleon2 supports two ways to specify checkpoint paths.

### Automatic Download from HuggingFace

Automatically downloads pre-trained checkpoints from HuggingFace:

```yaml
# In config files
ldm_module:
    vae_ckpt_path: ${hub:mp_20_vae}
    ldm_ckpt_path: ${hub:mp_20_ldm_base}
```

```bash
# In CLI
python src/train_ldm.py ldm_module.vae_ckpt_path='${hub:mp_20_vae}'
```

:::{tip} Available Pre-trained Checkpoints on HuggingFace
| Hub Key | Dataset | Description |
|---------|---------|-------------|
| `mp_20_vae` | MP-20 | pre-trained VAE|
| `alex_mp_20_vae` | Alex-MP-20 | pre-trained VAE|
| `mp_20_ldm_base` | MP-20 | pre-trained LDM|
| `alex_mp_20_ldm_base` | Alex-MP-20 | pre-trained LDM|
| `mp_20_ldm_rl`| MP-20 | Fine-tuned LDM with RL for DNG rewards|
| `alex_mp_20_ldm_rl` | Alex-MP-20 | Fine-tuned LDM with RL for DNG rewards|
:::

### Local File Paths

Use existing checkpoint files on your system:

```yaml
# In config files
ldm_module:
    vae_ckpt_path: ckpts/mp_20/vae/my_checkpoint.ckpt
```

```bash
# In CLI
python src/train_ldm.py ldm_module.vae_ckpt_path=ckpts/my_vae.ckpt
```

### Where Checkpoints Are Saved

During training, checkpoints are automatically saved to:
```
logs/{task}/runs/{timestamp}/checkpoints/
```

Examples:
- `logs/train_vae/runs/2025-11-02_09-35-59/checkpoints/`
- `logs/train_ldm/runs/2025-11-05_14-22-31/checkpoints/`
- `logs/train_rl/runs/2025-11-10_08-15-42/checkpoints/`

PyTorch Lightning's ModelCheckpoint callback (configured in [`configs/callbacks/default.yaml`](https://github.com/hspark1212/chemeleon2/blob/main/configs/callbacks/default.yaml)) saves:
- `last.ckpt`: Most recent (or last) epoch
- `epoch_*.ckpt`: Best checkpoints based on validation metrics

## Experiment Tracking

Chemeleon2 uses Weights & Biases (wandb) for logging by default.

### Setup

```bash
# First time: login to wandb
wandb login
```

### Offline Mode

Run without internet connection:

```bash
WANDB_MODE=offline python src/train_vae.py experiment=mp_20/vae_dng
```

### Custom Project/Run Names

```bash
python src/train_vae.py logger.wandb.project=my_project logger.wandb.name=my_run
```

## Next Steps

- [VAE Training](vae.md) - First stage: encode crystals to latent space
- [LDM Training](ldm.md) - Second stage: diffusion model in latent space
- [RL Training](rl.md) - Third stage: fine-tune with rewards
- [Predictor Training](predictor.md) - Optional: property prediction
