"""Reusable positional embedding components for VAEs."""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn


class PositionalEmbedding(nn.Module):
    """Base interface for positional embeddings."""

    def forward(self, indices: Tensor, emb_dim: int) -> Tensor:  # pragma: no cover
        raise NotImplementedError


class SinusoidalPositionalEmbedding(PositionalEmbedding):
    """Deterministic sine/cosine encoding (default)."""

    def __init__(self, max_len: int = 2048) -> None:
        super().__init__()
        self.max_len = max_len

    def forward(self, indices: Tensor, emb_dim: int) -> Tensor:
        if emb_dim % 2:
            raise ValueError("Sinusoidal positional embeddings require even emb_dim.")
        device = indices.device
        indices = indices.to(torch.float32)
        k = torch.arange(emb_dim // 2, device=device, dtype=torch.float32)
        div_term = torch.pow(self.max_len, 2 * k / emb_dim)
        angles = indices[..., None] * math.pi / div_term
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class LearnedPositionalEmbedding(PositionalEmbedding):
    """Simple learnable lookup table."""

    def __init__(self, embedding_dim: int, max_len: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, embedding_dim)

    def forward(self, indices: Tensor, emb_dim: int) -> Tensor:
        if emb_dim != self.embedding.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding.embedding_dim}, got {emb_dim}"
            )
        return self.embedding(indices)


class NoPositionalEmbedding(PositionalEmbedding):
    """Returns zeros to disable positional information."""

    def forward(self, indices: Tensor, emb_dim: int) -> Tensor:
        return torch.zeros((*indices.shape, emb_dim), device=indices.device)


class GlobalNumAtomsEmbedding(nn.Module):
    """
    Embed a per-graph scalar (number of atoms) and broadcast it to all nodes.

    Args:
        embedding_dim: Output feature dimension.
        mode: "learned" (nn.Embedding) or "sinusoidal".
        max_value: Max integer value expected for num_atoms (only for "learned").
        sinusoidal_max_len: Base used by the sinusoidal encoder (only for "sinusoidal").

    Shapes:
        num_atoms: Long tensor, shape (B,) with number of atoms per graph.
        num_nodes_per_graph: Long tensor, shape (B,) with node counts per graph.

    Returns:
        Tensor of shape (sum(num_nodes_per_graph), embedding_dim), containing the
        same embedding repeated for all nodes belonging to the same graph.
    """

    def __init__(
        self,
        embedding_dim: int,
        mode: str = "sinusoidal",
        max_value: int = 8192,
        sinusoidal_max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mode = mode.lower()
        if self.mode == "learned":
            # +1 so value==max_value is in-bounds
            self.table = nn.Embedding(max_value + 1, embedding_dim)
        elif self.mode == "sinusoidal":
            # Reuse the sinusoidal positional embedding defined above
            self.sine = SinusoidalPositionalEmbedding(max_len=sinusoidal_max_len)
            if embedding_dim % 2 != 0:
                raise ValueError("Sinusoidal mode requires an even embedding_dim.")
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'learned' or 'sinusoidal'.")

    def forward(self, num_atoms: Tensor, num_nodes_per_graph: Tensor) -> Tensor:
        if num_atoms.dim() != 1 or num_nodes_per_graph.dim() != 1:
            raise ValueError("num_atoms and num_nodes_per_graph must be 1D tensors.")
        if num_atoms.shape[0] != num_nodes_per_graph.shape[0]:
            raise ValueError("num_atoms and num_nodes_per_graph must have same length (B).")

        num_atoms = num_atoms.to(dtype=torch.long, device=num_nodes_per_graph.device)
        num_nodes_per_graph = num_nodes_per_graph.to(dtype=torch.long)

        # Broadcast per-graph scalar to per-node indices: [10, 4, ...] -> [10,10,...,4,4,...]
        per_node_indices = torch.repeat_interleave(num_atoms, num_nodes_per_graph)

        if self.mode == "learned":
            max_idx = int(per_node_indices.max().item()) if per_node_indices.numel() else -1
            if max_idx >= self.table.num_embeddings:
                raise ValueError(
                    f"num_atoms contains value {max_idx}, but embedding table "
                    f"has only {self.table.num_embeddings} entries. "
                    "Increase max_value when constructing GlobalNumAtomsEmbedding."
                )
            return self.table(per_node_indices)  # (N_total, embedding_dim)

        # Sinusoidal path
        return self.sine(per_node_indices, self.embedding_dim)  # (N_total, embedding_dim)

import torch
from torch import nn


import torch
from torch import nn

class SmoothEmbedding(nn.Module):
    """
    Smooth embedding: (..., 256) -> (..., embedding_dim)
    SiLU/GELU/RELU supported. Uses Kaiming for hidden layers (good proxy for SiLU/GELU),
    and Xavier for the output layer.
    """
    def __init__(
        self,
        input_dim: int = 256,
        embedding_dim: int = 128,
        width: int = 512,
        depth: int = 2,
        dropout: float = 0.0,
        activation: str = "silu",  # 'silu' | 'gelu' | 'relu' | 'tanh'
        use_layernorm: bool = True,
    ):
        super().__init__()
        act_map = {
            "silu": nn.SiLU(),
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported activation '{activation}'")
        self.activation_name = activation
        act = act_map[activation]

        layers = []
        if use_layernorm:
            layers.append(nn.LayerNorm(input_dim))

        in_d = input_dim
        for _ in range(depth):
            layers += [
                nn.Linear(in_d, width),
                act,
                nn.Dropout(dropout),
            ]
            in_d = width

        layers.append(nn.Linear(in_d, embedding_dim))  # output layer
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        # Hidden layers: Kaiming (good for ReLU/SiLU/GELU)
        for i, m in enumerate(self.net):
            if isinstance(m, nn.Linear):
                is_last = (i == len(self.net) - 1)
                if is_last:
                    # Output layer: Xavier with gain=1.0
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                else:
                    if self.activation_name in {"relu", "silu", "gelu"}:
                        nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                    elif self.activation_name == "tanh":
                        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
                    else:
                        nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class LinearEmbedding(nn.Module):
    """
    Simple linear map: (..., 256) -> (..., embedding_dim)
    Applies a single nn.Linear to the last dimension.
    """
    def __init__(self, input_dim: int = 256, embedding_dim: int = 128, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(input_dim, embedding_dim, bias=bias)
        # Stable default init
        nn.init.xavier_uniform_(self.proj.weight)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


__all__ = [
    "PositionalEmbedding",
    "SinusoidalPositionalEmbedding",
    "LearnedPositionalEmbedding",
    "NoPositionalEmbedding",
    "GlobalNumAtomsEmbedding",
    "SmoothEmbedding",
    "LinearEmbedding",
]
