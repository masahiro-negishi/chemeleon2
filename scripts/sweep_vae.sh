python -u src/sweep_vae.py experiment=train_vae
python -u src/sweep_vae.py experiment=train_lr_plateau
# Sweep for VAE with Crystal Transformer
python src/sweep_vae.py experiment=crystal_transformer/mp_20/sweep_vae_dng