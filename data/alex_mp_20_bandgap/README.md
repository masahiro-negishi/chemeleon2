# Finetuning Dataset Preparation Guide

## 1. Example Finetuning Dataset

This guide provides a step-by-step process to create a finetuning dataset using the `alex_mp_20` dataset from Microsoft MatterGen.
Create the following proceduer:

```python
import pandas as pd

# Load datasets from Microsoft MatterGen alex_mp_20
df_train = pd.read_csv("data/alex_mp_20/train.csv")
df_val = pd.read_csv("data/alex_mp_20/val.csv")
df_train = df_train.dropna(subset=["dft_band_gap"])
df_val = df_val.dropna(subset=["dft_band_gap"])
```

The finetuning example files are located in `data/finetuning/` directory, which contains the processed training and validation datasets.

## 2. Create Custom Finetuning Dataset

The required columns for the finetuning dataset are:

- `cif`: Convert pymatgen `Structure` objects to CIF format using the `to(fmt="cif")` method.

- target conditioning property (e.g., `dft_band_gap`, `formation_energy`, `dft_bulk_modulus`, `composition`, `chemical_system`)
