"""Graph construction utilities for crystal structures with periodic boundary conditions."""

import numpy as np
import torch
from ase import Atoms
from matscipy.neighbours import neighbour_list


def radius_graph_pbc(
    atoms: Atoms,
    r_cutoff: float = 6.0,
    max_neighbors: int | None = None,
    compute_edge_vectors: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute radius graph with PBC."""
    edge_src, edge_dst, distance, unit_shifts = neighbour_list(
        quantities="ijdS",
        atoms=atoms,
        cutoff=r_cutoff,
    )

    # Filter with max_neighbors if max_neighbors is not None
    if max_neighbors is not None:
        new_edge_src = []
        new_edge_dst = []
        new_edge_shift = []
        for i in range(len(atoms)):
            idx = np.where(edge_src == i)[0]
            sorted_idx = idx[np.argsort(distance[idx])][:max_neighbors]
            new_edge_src.extend(edge_src[sorted_idx])
            new_edge_dst.extend(edge_dst[sorted_idx])
            new_edge_shift.extend(unit_shifts[sorted_idx])
        edge_src = np.array(new_edge_src)
        edge_dst = np.array(new_edge_dst)
        unit_shifts = np.array(new_edge_shift)
        if len(edge_src) == 0:
            unit_shifts = np.empty((0, 3))

    # Construct the graph data
    cell = np.array(atoms.cell)
    edge_index = np.stack([edge_src, edge_dst], axis=0)
    shifts = np.dot(unit_shifts, cell)

    graph_data = {
        "edge_index": torch.as_tensor(edge_index, dtype=torch.long),
        "edge_shifts": torch.as_tensor(shifts, dtype=torch.float),
        "edge_unit_shifts": torch.as_tensor(unit_shifts, dtype=torch.float),
    }

    if compute_edge_vectors:
        pos = atoms.get_positions()
        edge_vectors = pos[edge_dst] + shifts - pos[edge_src]
        edge_distances = np.linalg.norm(edge_vectors, axis=1)
        graph_data["edge_vectors"] = torch.as_tensor(edge_vectors, dtype=torch.float)
        graph_data["edge_distances"] = torch.as_tensor(
            edge_distances, dtype=torch.float
        )

    return graph_data
