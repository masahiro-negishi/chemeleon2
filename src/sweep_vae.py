"""Hyperparameter sweep script for VAE training."""

from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from src.train_vae import main as run_train_vae


def run_sweep(cfg: DictConfig) -> None:
    with wandb.init() as run:
        # Deepcopy the original config
        _cfg = cfg.copy()

        # Log the hyperparameters
        wandb_config = run.config
        print(f"Running with config: {wandb_config}")

        # Update the config with wandb hyperparameters
        for key, value in wandb_config.items():
            OmegaConf.update(_cfg, key, value)

        # Call the training function with the updated config
        run_train_vae(_cfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="sweep_vae.yaml")
def main(cfg: DictConfig) -> None:
    # Get project name from config (default to "sweep-vae" if not specified)
    project = cfg.get("project", "sweep-vae")

    # Check if resuming an existing sweep
    if cfg.resume_sweep_id is not None:
        print(f"Resuming sweep with ID: {cfg.resume_sweep_id}")
        sweep_id = cfg.resume_sweep_id
    else:
        # Convert sweep config to container for wandb
        sweep_config = OmegaConf.to_container(cfg.sweep_config, resolve=True)
        assert isinstance(sweep_config, dict)

        # Initialize a new sweep
        sweep_id = wandb.sweep(sweep_config, project=project)
        print(f"Created new sweep with ID: {sweep_id}")

    # Start the sweep agent (with project specified for resume)
    wandb.agent(
        sweep_id,
        function=partial(run_sweep, cfg=cfg),
        count=cfg.sweep_count,
        project=project,
    )


if __name__ == "__main__":
    main()
