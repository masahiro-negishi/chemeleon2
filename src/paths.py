"""Default checkpoint paths for the project."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Default checkpoint paths
DEFAULT_VAE_CKPT_PATH = ROOT_DIR / "ckpts/alex_mp_20/vae/dng_j1jgz9t0_v1.ckpt"
DEFAULT_LDM_CKPT_PATH = ROOT_DIR / "ckpts/alex_mp_20/ldm/ldm_rl_dng_tuor5vgd.ckpt"
