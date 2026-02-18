"""Shared utilities for VAE encoders and decoders."""

import math

import torch


def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine/cosine positional embeddings from prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: dimension of the embeddings to create
        max_len: maximum length

    Returns:
        positional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
    return pos_embedding
