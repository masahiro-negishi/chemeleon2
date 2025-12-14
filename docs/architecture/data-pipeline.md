# Data Pipeline

The data module ([`src/data/`](https://github.com/hspark1212/chemeleon2/tree/main/src/data)) handles loading, processing, and batching of crystal structure data.

## Data Flow

```{mermaid}
flowchart LR
    A[CIF/JSON Files]
    B[Dataset]
    C[Featurizer]
    D[CrystalBatch]
    E[DataLoader]
    F[Model]

    A --> B --> C --> D --> E --> F

    style D fill:#e6f3ff
```

## Key Classes

### CrystalBatch

The core data container for batched crystal structures ([`src/data/schema.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/data/schema.py)):

```python
from src.data.schema import CrystalBatch

# CrystalBatch contains:
# - atom_types: Tensor of atomic numbers
# - frac_coords: Fractional coordinates
# - lengths: Lattice vector lengths
# - angles: Lattice angles
# - num_atoms: Number of atoms per structure
# - batch: Batch indices
```

### DataModule

PyTorch Lightning DataModule for training ([`src/data/datamodule.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/data/datamodule.py)):

```python
from src.data import DataModule

datamodule = DataModule(
    data_dir="data/mp-20",
    batch_size=32,
    num_workers=4,
)
```

### Featurizer

Converts pymatgen structures to model-ready features ([`src/utils/featurizer.py`](https://github.com/hspark1212/chemeleon2/blob/main/src/utils/featurizer.py)):

```python
from src.utils.featurizer import featurize

# Featurize structures
features = featurize(
    structures=[structure],
    model_path=None,  # Uses default pre-trained VAE
    batch_size=2000,
)
# Returns dict with "structure_features", "composition_features", "atom_features"
```

## Configuration

See [`configs/data/`](https://github.com/hspark1212/chemeleon2/tree/main/configs/data) for data configurations:

```yaml
# configs/data/mp-20.yaml (default)
_target_: src.data.datamodule.DataModule
data_dir: ${paths.data_dir}/mp-20
batch_size: 256
dataset_type: "mp_20"
num_workers: 16
```

:::{tip}
**Custom Datasets:** For creating custom datasets with property labels (e.g., band gap, formation energy), see the [Predictor-Based Reward Tutorial](../user-guide/rewards/predictor-reward.md#step-1-prepare-your-dataset).
:::

## Learn More

- [Predictor-Based Reward Tutorial](../user-guide/rewards/predictor-reward.md) - Guide on using predictors for RL rewards
- [API Reference](../api/index.md) - Full API documentation
