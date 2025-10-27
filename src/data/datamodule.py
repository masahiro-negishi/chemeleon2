"""Lightning DataModule for crystal structure datasets.

This module provides a LightningDataModule wrapper for loading and batching
crystal structure data with support for Materials Project datasets.
"""

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from src.data.schema import CrystalBatch


class DataModule(LightningDataModule):
    """PyTorch Lightning DataModule for crystal structure datasets."""

    def __init__(
        self,
        dataset: DictConfig,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__()
        # Configs for dataset
        self.dataset_config = dataset
        self.name = name

        # Configs for dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = instantiate(self.dataset_config, split="train")
            self.val_dataset = instantiate(self.dataset_config, split="val")
        if stage == "test" or stage is None:
            self.test_dataset = instantiate(self.dataset_config, split="test")

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        loader.collate_fn = CrystalBatch.collate
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        loader.collate_fn = CrystalBatch.collate
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        loader.collate_fn = CrystalBatch.collate
        return loader
