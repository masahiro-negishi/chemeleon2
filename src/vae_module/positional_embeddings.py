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


__all__ = [
    "PositionalEmbedding",
    "SinusoidalPositionalEmbedding",
    "LearnedPositionalEmbedding",
    "NoPositionalEmbedding",
]
