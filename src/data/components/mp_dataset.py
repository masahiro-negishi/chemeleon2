"""Materials Project dataset implementation for crystal structure data."""

import os
import warnings
from collections.abc import Callable, Iterable

import h5py
import pandas as pd
import torch
from ase import Atoms
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from src.data.dataset_util import pmg_structure_to_pyg_data
from src.data.graph_utils import radius_graph_pbc

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")


class MPDataset(InMemoryDataset):
    """InMemoryDataset for Materials Project data that caches processed graphs."""

    def __init__(
        self,
        root: str,
        split: str,
        compute_radius_graph: bool = False,
        r_cutoff: float = 6.0,
        max_neighbors: int | None = None,
        compute_edge_vectors: bool = True,
        target_condition: str | Iterable[str] | None = None,
        mace_features: bool = False,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
    ) -> None:
        """Initialize Materials Project dataset.

        Args:
            root: Root directory containing dataset files.
            split: Dataset split name (train/val/test).
            compute_radius_graph: Whether to compute radius graph with PBC.
            r_cutoff: Maximum radius (cutoff) for graph edges.
            max_neighbors: Maximum number of neighbors per atom (None for no limit).
            compute_edge_vectors: Whether to compute edge vectors and distances.
            target_condition: Optional target property for conditioning.
            mace_features: Whether to load MACE structural features.
            transform: Optional transform to apply on-the-fly.
            pre_transform: Optional transform to apply during processing.
        """
        self.split = split
        self.r_cutoff = r_cutoff
        self.compute_radius_graph = compute_radius_graph
        self.max_neighbors = max_neighbors
        self.compute_edge_vectors = compute_edge_vectors
        self.target_condition = target_condition
        self.mace_features = mace_features

        # Load raw DataFrame for dynamic condition lookup
        raw_path = os.path.join(root, f"{self.split}.csv")
        self.df = pd.read_csv(raw_path).reset_index(drop=True)

        super().__init__(root, transform, pre_transform)

        # Load processed data
        data_path = self.processed_paths[0]
        self.data, self.slices = torch.load(data_path, weights_only=False)

        # Optionally add MACE features
        if mace_features:
            self.mace_features_dict = dict()
            with h5py.File(os.path.join(root, "mace_features.h5"), "r") as f:
                for material_id in self.df["material_id"]:
                    if str(material_id) in f:
                        self.mace_features_dict[material_id] = torch.tensor(
                            f[str(material_id)][:]  # type: ignore[index]
                        )

    @property
    def raw_file_names(self) -> list[str]:
        """Return list of raw file names."""
        return [f"{self.split}.csv"]

    @property
    def raw_paths(self) -> list[str]:
        """Return full paths to raw files."""
        return [os.path.join(self.root, f) for f in self.raw_file_names]

    @property
    def processed_file_names(self) -> list[str]:
        """Return list of processed file names."""
        return [f"{self.split}.pt"]

    def download(self) -> None:
        """Download dataset (placeholder - data should be manually placed)."""
        # download https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20
        return

    def process(self) -> None:
        """Process raw data files into PyG Data objects."""
        data_list: list[Data] = []
        for _, row in tqdm(
            self.df.iterrows(),
            total=len(self.df),
            desc=f"Processing {self.split} dataset",
        ):
            # Get material ID
            material_id = row["material_id"]

            # Parse CIF string
            cif_str = str(row["cif"])
            st = Structure.from_str(cif_str, fmt="cif")  # type: ignore[arg-type]

            # Niggli reduction for canonical form
            reduced = st.get_reduced_structure()
            canonical = Structure(
                lattice=Lattice.from_parameters(*reduced.lattice.parameters),
                species=reduced.species,
                coords=reduced.frac_coords,
                coords_are_cartesian=False,
            )

            # Convert to PyG Data
            graph = pmg_structure_to_pyg_data(canonical, material_id=material_id)
            data_list.append(graph)

        # Optionally apply pre_transform
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx: int):
        """Get data object by index with optional conditions and MACE features.

        Args:
            idx: Index of the data sample.

        Returns:
            PyG Data object with structure and optional conditions/features.
        """
        data = super().get(idx)

        # Dynamically compute radius graph if specified
        if self.compute_radius_graph:
            atoms = Atoms(
                numbers=data.atom_types.detach().cpu().numpy(),
                cell=data.lattices.squeeze(0).detach().cpu().numpy(),
                positions=data.cart_coords.detach().cpu().numpy(),
                pbc=True,
            )
            edge_graph = radius_graph_pbc(
                atoms=atoms,
                r_cutoff=self.r_cutoff,
                max_neighbors=self.max_neighbors,
                compute_edge_vectors=self.compute_edge_vectors,
            )
            data.update(edge_graph)  # type: ignore

        # Dynamically attach condition if specified
        if self.target_condition is not None:
            if isinstance(self.target_condition, str):  # single condition
                if self.target_condition not in self.df.columns:
                    msg = f"Condition {self.target_condition} not in dataframe columns"
                    raise ValueError(msg)
                y = {self.target_condition: self.df.loc[idx, self.target_condition]}
            elif isinstance(self.target_condition, Iterable):  # multiple conditions
                if not all(t in self.df.columns for t in self.target_condition):
                    msg = "Not all conditions found in dataframe columns"
                    raise ValueError(msg)
                y = {t: self.df.loc[idx, t] for t in self.target_condition}
            else:
                raise ValueError("target_condition must be str or iterable[str]")
            data.target_condition = self.target_condition
            data.y = y

        # Dynamically attach MACE features if available
        if self.mace_features:
            material_id = self.df.loc[idx, "material_id"]
            data.mace_features = self.mace_features_dict[material_id]
        return data
