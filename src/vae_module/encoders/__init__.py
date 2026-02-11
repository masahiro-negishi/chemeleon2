"""Encoder architectures for VAE module."""

from src.vae_module.encoders.cspnet import CSPNet
from src.vae_module.encoders.precomputed_mace import PrecomputedMACEEncoder
from src.vae_module.encoders.transformer import TransformerEncoder

__all__ = ["CSPNet", "PrecomputedMACEEncoder", "TransformerEncoder"]

# Lazy import for MACEEncoder (requires optional 'training' extra)
try:
    from src.vae_module.encoders.mace import MACEEncoder

    __all__.append("MACEEncoder")
except ImportError:
    # MACE not installed - MACEEncoder unavailable
    # Install with: uv sync --extra training
    pass
