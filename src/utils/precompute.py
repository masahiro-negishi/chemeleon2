"""Utilities for automatic MACE embedding pre-computation."""

from pathlib import Path

import h5py
import pandas as pd
import torch


def validate_embeddings(
    data_dir: Path, expected_hidden_dim: int
) -> tuple[bool, str]:
    """Validate if pre-computed embeddings exist and are complete.

    Args:
        data_dir: Path to data directory containing mace_embeddings.h5 and CSV splits
        expected_hidden_dim: Expected embedding dimension for validation

    Returns:
        Tuple of (is_valid, reason) where reason explains why if invalid
    """
    embeddings_path = data_dir / "mace_embeddings.h5"

    # Check if file exists
    if not embeddings_path.exists():
        return False, "mace_embeddings.h5 does not exist"

    try:
        # Open HDF5 file and check structure
        with h5py.File(embeddings_path, "r") as f:
            # Collect all material IDs that should be in the file
            required_ids = set()
            for split in ["train", "val", "test"]:
                csv_path = data_dir / f"{split}.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    required_ids.update(df["material_id"].astype(str).tolist())

            if not required_ids:
                return False, "No CSV files found in data directory"

            # Check if all required IDs are present
            missing_ids = required_ids - set(f.keys())
            if missing_ids:
                return (
                    False,
                    f"Missing embeddings for {len(missing_ids)} material IDs",
                )

            # Validate dimension on first embedding
            first_id = next(iter(required_ids))
            embedding = f[first_id][:]
            if embedding.shape[1] != expected_hidden_dim:
                return (
                    False,
                    f"Dimension mismatch: expected {expected_hidden_dim}, got {embedding.shape[1]}",
                )

        return True, "Valid"

    except Exception as e:
        return False, f"Error validating embeddings: {e!s}"


def run_precompute_for_split(
    data_dir: Path,
    split: str,
    model: str,
    head: str,
    device: str,
    output_name: str = "mace_embeddings.h5",
) -> None:
    """Pre-compute MACE embeddings for a single data split.

    Directly integrates the precomputing workflow using MACEEncoder,
    avoiding subprocess overhead and providing better integration.

    Args:
        data_dir: Path to data directory
        split: Data split name (train/val/test)
        model: MACE model name
        head: MACE head name for multihead models
        device: Compute device (cuda/cpu)
        output_name: Output HDF5 filename

    Raises:
        RuntimeError: If pre-computation fails
    """
    from tqdm import tqdm

    from src.data.components.mp_dataset import MPDataset
    from src.data.schema import CrystalBatch
    from src.vae_module.encoders.mace import MACEEncoder

    print(f"→ Pre-computing embeddings for {split} split...")
    print(f"  Data directory: {data_dir}")
    print(f"  Device: {device}")

    # Load dataset (without MACE embeddings, we're creating them)
    print(f"  Loading {split} dataset...")
    dataset = MPDataset(
        root=str(data_dir),
        split=split,
        mace_embeddings=False,
        mace_features=False,
    )
    print(f"  Loaded {len(dataset)} structures")

    # Initialize encoder
    print(f"  Initializing MACEEncoder...")
    print(f"    Model: {model}")
    print(f"    Head: {head}")

    encoder = MACEEncoder(
        model=model,
        head=head,
    )
    encoder = encoder.to(device)
    encoder.eval()

    print(f"    Hidden dimension: {encoder.hidden_dim}")

    # Pre-compute embeddings
    print(f"  Pre-computing embeddings...")

    output_path = data_dir / output_name

    # Create or append to HDF5 file
    mode = "a" if output_path.exists() else "w"
    with h5py.File(output_path, mode) as h5f:
        for idx in tqdm(range(len(dataset)), desc="  Processing structures"):
            # Get single structure
            data = dataset[idx]
            material_id = str(data.material_id)

            # Skip if already exists
            if material_id in h5f:
                continue

            # Create single-item batch
            batch = CrystalBatch.collate([data])
            batch = batch.to(device)

            # Forward pass (no gradients needed)
            with torch.no_grad():
                output = encoder(batch)

            # Extract embeddings (per-atom features)
            embeddings = output["x"].cpu().numpy()

            # Store in HDF5 with material_id as key
            h5f.create_dataset(
                material_id,
                data=embeddings.astype("float32"),
                compression="gzip",
                compression_opts=4,
            )

    print(f"  Successfully saved embeddings to {output_path}")
    print(f"✓ {split} split completed\n")


def ensure_mace_embeddings(data_dir: Path, params: dict) -> None:
    """Ensure MACE embeddings exist, pre-computing if necessary.

    This is the main orchestrator function that checks if embeddings exist
    and are valid, and triggers pre-computation if needed.

    Args:
        data_dir: Path to data directory
        params: Dictionary with keys:
            - model: MACE model name (required)
            - head: MACE head name (optional, default "Default")
            - hidden_dim: Expected embedding dimension (required)
            - device: Compute device (optional, auto-detect if not provided)

    Raises:
        ValueError: If required parameters are missing
        RuntimeError: If pre-computation fails
    """
    # Validate required parameters
    required_keys = ["model", "hidden_dim"]
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise ValueError(
            f"Missing required parameters in mace_precompute_params: {missing_keys}"
        )

    # Extract parameters with defaults
    model = params["model"]
    head = params.get("head", "Default")
    hidden_dim = params["hidden_dim"]
    device = params.get("device")

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nChecking MACE embeddings...")
    print(f"  Data directory: {data_dir}")
    print(f"  Model: {model}, Head: {head}")

    # Check if embeddings already exist and are valid
    is_valid, reason = validate_embeddings(data_dir, hidden_dim)

    if is_valid:
        print("✓ Embeddings already exist and are valid\n")
        return

    print(f"✗ Embeddings need pre-computation: {reason}\n")
    print("Starting automatic pre-computation...")
    print("This will take 1-2 hours. Progress will be shown below.\n")

    # Pre-compute for each split
    for split in ["train", "val", "test"]:
        run_precompute_for_split(
            data_dir=data_dir,
            split=split,
            model=model,
            head=head,
            device=device,
        )

    # Validate result
    is_valid, reason = validate_embeddings(data_dir, hidden_dim)
    if not is_valid:
        raise RuntimeError(
            f"Pre-computation completed but validation failed: {reason}"
        )

    print("✓ All embeddings pre-computed successfully!\n")
