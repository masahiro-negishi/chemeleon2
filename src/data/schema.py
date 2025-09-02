from ase import Atoms
from pymatgen.core import Structure

import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from src.data.dataset_util import batch_to_atoms_list, batch_to_structure_list


class CrystalBatch(Batch):

    def add(self, **kwargs) -> None:
        for key, tensor in kwargs.items():
            if not isinstance(key, str):
                raise TypeError(f"Key must be a string, got {type(key).__name__}.")
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    f"Value must be a torch.Tensor, got {type(tensor).__name__}."
                )
            if hasattr(self, key):
                raise KeyError(f"Attribute '{key}' already exists in the batch.")
            setattr(self, key, tensor)

    def update(self, allow_reshape: bool = False, **kwargs) -> None:
        for key, tensor in kwargs.items():
            if not isinstance(key, str):
                raise TypeError(f"Key must be a string, got {type(key).__name__}.")
            if not hasattr(self, key):
                raise KeyError(f"Attribute '{key}' not found in the batch.")
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    f"Value must be a torch.Tensor, got {type(tensor).__name__}."
                )

            existing = getattr(self, key)
            if tensor.shape != existing.shape and not allow_reshape:
                raise ValueError(
                    f"Shape mismatch for '{key}': existing {tuple(existing.shape)}, new {tuple(tensor.shape)}."
                )
            setattr(self, key, tensor)

    def remove(self, *keys: str) -> None:
        if len(keys) == 0:
            raise ValueError("At least one key must be provided to remove().")
        for key in keys:
            if not isinstance(key, str):
                raise TypeError(f"Key must be a string, got {type(key).__name__}.")
            if not hasattr(self, key):
                raise KeyError(f"Attribute '{key}' not found in the batch.")
            delattr(self, key)

    def to(self, device: any, non_blocking: bool = False) -> "CrystalBatch":
        return self.apply(lambda x: x.to(device, non_blocking=non_blocking))

    def to_atoms(self, **kwargs) -> list[Atoms]:
        return batch_to_atoms_list(self, **kwargs)

    def to_structure(self, **kwargs) -> list[Structure]:
        return batch_to_structure_list(self, **kwargs)

    def repeat(self, num_repeats: int) -> "CrystalBatch":
        return super().from_data_list(
            [d.clone() for d in self.to_data_list() for _ in range(num_repeats)]
        )

    @classmethod
    def collate(cls, data_list: list[any]) -> "CrystalBatch":
        batch = super(CrystalBatch, cls).from_data_list(data_list)
        batch.__class__ = cls
        return batch


def create_empty_batch(
    num_atoms: list[int],
    device: str,
    atom_types: list[list[int]] | None = None,
) -> CrystalBatch:
    return CrystalBatch.from_data_list(
        [
            Data(
                pos=torch.empty((n, 3)),
                atom_types=(
                    torch.empty((n,))
                    if atom_types is None
                    else torch.tensor(atom_types[i], dtype=torch.long)
                ),
                frac_coords=torch.empty((n, 3)),
                cart_coords=torch.empty((n, 3)),
                lattices=torch.empty((1, 3, 3)),
                num_atoms=torch.as_tensor(n, dtype=torch.long),
                lengths=torch.empty((1, 3)),
                lengths_scaled=torch.empty((1, 3)),
                angles=torch.empty((1, 3)),
                angles_radians=torch.empty((1, 3)),
                token_idx=torch.arange(n, dtype=torch.long),
            )
            for i, n in enumerate(num_atoms)
        ]
    ).to(device=device)
