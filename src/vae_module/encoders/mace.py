"""Frozen MACE foundation model encoder for VAE."""

import numpy as np
import torch
from mace.data.neighborhood import get_neighborhood
from mace.modules.utils import extract_invariant
from mace.tools import AtomicNumberTable, atomic_numbers_to_indices
from torch import Tensor, nn

from src.data.schema import CrystalBatch


class MACEEncoder(nn.Module):
    """Frozen MACE encoder that extracts per-atom descriptors.

    Uses a pre-trained MACE foundation model to produce fixed,
    physicochemically-informed per-atom features. The MACE model
    is kept frozen — only downstream layers (quant_conv, decoder)
    are trained.

    Args:
        model: MACE-MP model name (e.g. "mh-1", "medium", "large")
            or a file path to a custom MACE checkpoint.
        head: Head name for multihead models (e.g. "omat_pbe").
        default_dtype: Float precision for the MACE calculator.
        max_num_elements: Maximum number of element types (for config compatibility).
            Ignored by MACE; actual value determined from model's z_table.
    """

    def __init__(
        self,
        model: str = "mh-1",
        head: str | None = "omat_pbe",
        default_dtype: str = "float64",
        max_num_elements: int = 100,
    ) -> None:
        super().__init__()
        from mace.calculators.foundations_models import mace_mp

        # Save and restore default dtype to avoid global state mutation
        original_dtype = torch.get_default_dtype()
        calc = mace_mp(
            model=model,
            device="cpu",
            default_dtype=default_dtype,
            head=head,
        )
        torch.set_default_dtype(original_dtype)

        self.mace_model: nn.Module = calc.models[0]
        self.mace_model.eval()
        for param in self.mace_model.parameters():
            param.requires_grad_(False)

        # Cache model metadata needed for batching
        self._r_max = float(self.mace_model.r_max)
        self._z_table: AtomicNumberTable = calc.z_table
        self._num_elements = len(self._z_table)

        # Head index for multi-head models
        available_heads = list(getattr(self.mace_model, "heads", ["Default"]))
        self._head_index = (
            available_heads.index(head) if head and head in available_heads else 0
        )

        # Determine descriptor dimension from the model.
        # get_descriptors() concatenates per-layer invariant features:
        #   each layer produces num_invariant_features scalars,
        #   total = num_invariant_features * num_interactions.
        from e3nn import o3

        irreps_out = o3.Irreps(str(self.mace_model.products[0].linear.irreps_out))
        self._l_max = irreps_out.lmax
        self._num_invariant_features = irreps_out.dim // (self._l_max + 1) ** 2
        self._num_interactions = int(self.mace_model.num_interactions)

        # Use only final layer's invariant features (most abstract representation).
        # Alternative: concatenate all layers for richer but higher-dimensional features.
        self._hidden_dim = self._num_invariant_features

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def max_num_elements(self) -> int:
        """Number of elements supported by the MACE model's z_table."""
        return self._num_elements

    def train(self, mode: bool = True) -> "MACEEncoder":
        """Override to keep MACE model always in eval mode."""
        super().train(mode)
        self.mace_model.eval()
        return self

    def _build_mace_batch(
        self, batch: CrystalBatch, device: torch.device
    ) -> dict[str, Tensor]:
        """Build batched MACE input dict from CrystalBatch.

        Args:
            batch: CrystalBatch with atom_types, frac_coords, lattices, etc.
            device: Target device for tensors.

        Returns:
            Dict with MACE model input tensors.
        """
        # Get model dtype
        model_dtype = next(self.mace_model.parameters()).dtype

        # Per-structure loop to build neighbor lists (CPU-bound)
        positions_list = []
        node_attrs_list = []
        shifts_list = []
        unit_shifts_list = []
        edge_index_list = []
        cell_list = []
        node_offset = 0

        num_structures = int(batch.batch.max().item()) + 1

        for i in range(num_structures):
            # Slice data for structure i
            mask = batch.batch == i
            atom_types_i = batch.atom_types[mask]
            frac_coords_i = batch.frac_coords[mask]
            lattice_i = batch.lattices[i]

            # Convert to Cartesian coordinates
            cart_coords_i = frac_coords_i @ lattice_i

            # Convert to numpy for neighborhood calculation
            positions_np = cart_coords_i.cpu().numpy()
            cell_np = lattice_i.cpu().numpy()

            # Get neighborhood (returns 4 values: edge_index, shifts, unit_shifts, cell)
            edge_index_np, shifts_np, unit_shifts_np, _ = get_neighborhood(
                positions=positions_np,
                cutoff=self._r_max,
                pbc=np.array([True, True, True]),
                cell=cell_np,
            )

            # Convert atomic numbers to indices and one-hot
            atomic_numbers = atom_types_i.cpu().numpy()
            indices = atomic_numbers_to_indices(atomic_numbers, z_table=self._z_table)
            node_attrs = torch.nn.functional.one_hot(
                torch.tensor(indices, dtype=torch.long),
                num_classes=self._num_elements,
            ).to(dtype=torch.float)

            # Store tensors (still on CPU)
            positions_list.append(torch.from_numpy(positions_np))
            node_attrs_list.append(node_attrs)
            shifts_list.append(torch.from_numpy(shifts_np))
            unit_shifts_list.append(torch.from_numpy(unit_shifts_np))

            # Offset edge indices for batching
            edge_index_offset = torch.from_numpy(edge_index_np) + node_offset
            edge_index_list.append(edge_index_offset)

            cell_list.append(torch.from_numpy(cell_np).unsqueeze(0))

            node_offset += len(atom_types_i)

        # Concatenate all structures
        positions = torch.cat(positions_list, dim=0).to(
            device=device, dtype=model_dtype
        )
        node_attrs = torch.cat(node_attrs_list, dim=0).to(
            device=device, dtype=model_dtype
        )
        shifts = torch.cat(shifts_list, dim=0).to(device=device, dtype=model_dtype)
        unit_shifts = torch.cat(unit_shifts_list, dim=0).to(
            device=device, dtype=torch.long
        )
        edge_index = torch.cat(edge_index_list, dim=1).to(
            device=device, dtype=torch.long
        )
        cell = torch.cat(cell_list, dim=0).to(device=device, dtype=model_dtype)

        # Build batch vector and ptr tensor
        batch_vec = batch.batch.to(device=device)
        ptr = torch.zeros(num_structures + 1, dtype=torch.long, device=device)
        for i in range(num_structures):
            ptr[i + 1] = ptr[i] + (batch.batch == i).sum()

        # Build head tensor
        head = torch.full(
            (num_structures,), self._head_index, dtype=torch.long, device=device
        )

        return {
            "positions": positions,
            "node_attrs": node_attrs,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "edge_index": edge_index,
            "cell": cell,
            "batch": batch_vec,
            "ptr": ptr,
            "head": head,
        }

    def forward(self, batch: CrystalBatch) -> dict[str, Tensor]:
        """Extract per-atom MACE descriptors for a crystal batch.

        Args:
            batch: CrystalBatch with atom_types, frac_coords, lattices, etc.

        Returns:
            Dict with keys "x", "num_atoms", "batch", "token_idx".
        """
        device = next(self.mace_model.parameters()).device
        data_dict = self._build_mace_batch(batch, device)

        with torch.no_grad():
            output = self.mace_model(
                data_dict,
                training=False,
                compute_force=False,
            )

        node_feats = output["node_feats"]

        # Extract invariant features from all layers, then select only the final layer
        # extract_invariant returns concatenated invariants: (num_atoms, num_layers * num_features)
        all_invariants = extract_invariant(
            node_feats,
            num_layers=self._num_interactions,
            num_features=self._num_invariant_features,
            l_max=self._l_max,
        )
        # Take only the final layer's features (most abstract representation)
        start = (self._num_interactions - 1) * self._num_invariant_features
        end = self._num_interactions * self._num_invariant_features
        x = all_invariants[:, start:end]

        return {
            "x": x.to(dtype=torch.float32),
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }
