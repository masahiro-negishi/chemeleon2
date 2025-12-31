"""Reward components for reinforcement learning."""

import gzip
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict

import amd
import numpy as np
import torch
from ElMD import ElMD
from huggingface_hub import hf_hub_download
from pymatgen.core import Structure
from xtalmet.constants import HF_VERSION
from xtalmet.crystal import Crystal

from src.utils.featurizer import featurize
from src.utils.metrics import Metrics, structures_to_amd

###############################################################################
#                              Reward Components                              #
###############################################################################


class RewardComponent(ABC, torch.nn.Module):
    """Base class for all reward components."""

    required_metrics: list[str] = []

    def __init__(
        self,
        weight: float = 1.0,
        normalize_fn: str | None = None,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.weight = weight
        self.normalize_fn = normalize_fn
        self.eps = eps

    @abstractmethod
    def compute(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the raw reward."""
        pass

    def forward(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """Compute and optionally normalize the reward."""
        rewards = self.compute(
            **kwargs,
        )
        if self.normalize_fn:
            rewards = self._normalize(rewards)

        if "device" in kwargs:
            rewards = rewards.to(kwargs["device"])

        return rewards * self.weight

    def _normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.normalize_fn == "norm":
            rewards = normalize(rewards, eps=self.eps)
        elif self.normalize_fn == "std":
            rewards = standardize(rewards, eps=self.eps)
        elif self.normalize_fn == "subtract_mean":
            rewards = rewards - rewards.mean()
        elif self.normalize_fn == "clip":
            rewards = torch.clamp(rewards, min=-1.0, max=1.0)
        elif self.normalize_fn is None:
            pass
        else:
            raise ValueError(
                f"Unknown normalization type: {self.normalize_fn}. Use 'norm', 'std', 'clip', or None."
            )
        return rewards


###############################################################################
#                           Custom Reward Components                          #
###############################################################################
class CustomReward(RewardComponent):
    """Wrapper for user-defined custom reward functions."""

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        """Placeholder for custom reward function."""
        return torch.zeros(len(gen_structures))


class BSUNReward(RewardComponent):
    """Binary SUN reward using StructureMatcher."""

    required_metrics = ["unique", "novel", "e_above_hull"]

    def compute(
        self, gen_structures: list[Structure], metrics_obj: Metrics, **kwargs
    ) -> torch.Tensor:
        return torch.as_tensor(metrics_obj._results["MSUN"]).float()


class CSUNReward(RewardComponent):
    """Continuous SUN reward using ElMD+AMD."""

    required_metrics = ["e_above_hull"]

    def __init__(
        self,
        weight: float = 1.0,
        normalize_fn: str | None = None,
        eps: float = 1e-4,
    ):
        super().__init__(
            weight=weight,
            normalize_fn=normalize_fn,
            eps=eps,
        )
        print("Downloading MP-20 training data from Hugging Face...")
        path_embs_train = hf_hub_download(
            repo_id="masahiro-negishi/xtalmet",
            filename="mp20/train/train_elmd+amd.pkl.gz",
            repo_type="dataset",
            revision=HF_VERSION,
        )
        with gzip.open(path_embs_train, "rb") as f:
            ref_embs = pickle.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ref_embs_elmd = torch.tensor(
            np.stack([ElMD(emb[0]).feature_vector for emb in ref_embs])
        ).to(device)
        self.ref_embs_amd = torch.tensor(np.stack([emb[1] for emb in ref_embs])).to(
            device
        )
        self.device = device

    def compute(
        self, gen_structures: list[Structure], metrics_obj: Metrics, **kwargs
    ) -> torch.Tensor:
        # embeddings
        gen_xtals = [Crystal.from_Structure(s) for s in gen_structures]
        gen_embs_elmd = torch.tensor(
            np.stack(
                [ElMD(xtal._get_emb_d_elmd()).feature_vector for xtal in gen_xtals]
            )
        ).to(self.device)
        gen_embs_amd = []
        error_indices = []
        for i, xtal in enumerate(gen_xtals):
            try:
                emb_amd = xtal._get_emb_d_amd()
                gen_embs_amd.append(emb_amd)
            except Exception:
                error_indices.append(i)
        gen_embs_amd = torch.tensor(np.stack(gen_embs_amd)).to(self.device)

        # coeff
        coef_elmd = float.fromhex("0x1.8d7d565a99f87p-1")
        coef_amd = float.fromhex("0x1.ca0aa695981e5p-3")

        # uniqueness
        d_mtx_uni_elmd = torch.sum(
            torch.abs(
                (gen_embs_elmd[:, None, :] - gen_embs_elmd[None, :, :]).cumsum(dim=-1)
            ),
            dim=-1,
        )
        d_mtx_uni_amd = torch.max(
            torch.abs(gen_embs_amd[:, None, :] - gen_embs_amd[None, :, :]), dim=-1
        )[0]
        for i in error_indices:
            d_mtx_uni_amd = torch.cat(
                [
                    d_mtx_uni_amd[:i, :],
                    torch.full((1, d_mtx_uni_amd.shape[1]), float("nan")).to(
                        self.device
                    ),
                    d_mtx_uni_amd[i:, :],
                ],
                dim=0,
            )
            d_mtx_uni_amd = torch.cat(
                [
                    d_mtx_uni_amd[:, :i],
                    torch.full((d_mtx_uni_amd.shape[0], 1), float("nan")).to(
                        self.device
                    ),
                    d_mtx_uni_amd[:, i:],
                ],
                dim=1,
            )
        d_mtx_uni = coef_elmd * d_mtx_uni_elmd / (
            1 + d_mtx_uni_elmd
        ) + coef_amd * d_mtx_uni_amd / (1 + d_mtx_uni_amd)
        uni_scores = torch.sum(d_mtx_uni, dim=1) / (len(gen_structures) - 1)
        uni_scores = uni_scores.cpu().numpy()

        # novelty
        d_mtx_nov_elmd = torch.sum(
            torch.abs(
                (gen_embs_elmd[:, None, :] - self.ref_embs_elmd[None, :, :]).cumsum(
                    dim=-1
                )
            ),
            dim=-1,
        )
        d_mtx_nov_amd = torch.max(
            torch.abs(gen_embs_amd[:, None, :] - self.ref_embs_amd[None, :, :]), dim=-1
        )[0]
        for i in error_indices:
            d_mtx_nov_amd = torch.cat(
                [
                    d_mtx_nov_amd[:i, :],
                    torch.full((1, d_mtx_nov_amd.shape[1]), float("nan")).to(
                        self.device
                    ),
                    d_mtx_nov_amd[i:, :],
                ],
                dim=0,
            )
        d_mtx_nov = coef_elmd * d_mtx_nov_elmd / (
            1 + d_mtx_nov_elmd
        ) + coef_amd * d_mtx_nov_amd / (1 + d_mtx_nov_amd)
        nov_scores = torch.min(d_mtx_nov, dim=1)[0]
        nov_scores = nov_scores.cpu().numpy()

        # stability
        stability_scores = np.zeros(len(gen_structures), dtype=float)
        isnan = np.isnan(metrics_obj._results["e_above_hull"])
        stability_scores[~isnan] = np.clip(
            1 - metrics_obj._results["e_above_hull"][~isnan] / 0.4289, 0, 1
        )
        scores = stability_scores * uni_scores * nov_scores
        scores[np.isnan(scores)] = 0.0
        return torch.from_numpy(scores).float()


###############################################################################
#                          Built-in Reward Components                         #
###############################################################################
class CreativityReward(RewardComponent):
    """Combined Unique and Novel reward with AMD distance fallback."""

    required_metrics = ["unique", "novel"]

    def compute(
        self,
        gen_structures: list[Structure],
        metrics_obj: Metrics,
        **kwargs,
    ) -> torch.Tensor:
        # Ensure metrics_obj is provided in kwargs
        reference_structures = metrics_obj._reference_structures
        metrics_results = metrics_obj._results

        # Build a mapping from formula to reference structures
        ref_structures_by_formula = defaultdict(list)
        for ref_structure in reference_structures + gen_structures:
            ref_structures_by_formula[ref_structure.reduced_formula].append(
                ref_structure
            )

        # Compute creativity rewards
        rewards = []
        for i, gen_structure in enumerate(gen_structures):
            u, v = metrics_results["unique"][i], metrics_results["novel"][i]
            if u and v:
                r = 1.0
            elif not u and not v:
                r = 0.0
            else:
                matching_refs = ref_structures_by_formula.get(
                    gen_structure.reduced_formula, []
                )
                amds = structures_to_amd([gen_structure] + matching_refs, 100)
                dists = amd.AMD_cdist(amds, amds)[0]
                r = dists[dists > 0].min()
            rewards.append(r)

        return torch.as_tensor(rewards).float()


class EnergyReward(RewardComponent):
    """Rewards structures with low energy above convex hull."""

    required_metrics = ["e_above_hull"]

    def compute(
        self,
        gen_structures: list[Structure],
        metrics_obj: Metrics,
        **kwargs,
    ) -> torch.Tensor:
        metrics_results = metrics_obj._results

        r_energy = torch.as_tensor(metrics_results["e_above_hull"]).float()
        r_energy = r_energy.nan_to_num(nan=1.0)  # max clip energy
        r_energy = r_energy.clamp(min=0.0, max=1.0)
        r_energy = r_energy * -1.0  # Negative for minimization
        return r_energy


class StructureDiversityReward(RewardComponent):
    """Rewards diverse crystal structures using MMD."""

    required_metrics = ["structure_diversity"]

    def compute(
        self,
        gen_structures: list[Structure],
        metrics_obj: Metrics,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        # Get reference structure features
        assert metrics_obj._reference_structure_features is not None
        ref_structure_features: torch.Tensor = metrics_obj._reference_structure_features

        # Get generated structure features
        gen_features = featurize(gen_structures)
        gen_structure_features = gen_features["structure_features"].to(device)
        ref_structure_features = ref_structure_features.to(device)
        if len(ref_structure_features) > 50000:  # Subsample for efficiency
            indices = torch.randperm(len(ref_structure_features))[:50000]
            ref_structure_features = ref_structure_features[indices]

        # Compute MMD reward
        r_structure_diversity = mmd_reward(
            z_gen=gen_structure_features, z_ref=ref_structure_features
        )["r_indiv"]
        return r_structure_diversity


class CompositionDiversityReward(RewardComponent):
    """Rewards diverse chemical compositions using MMD."""

    required_metrics = ["composition_diversity"]

    def compute(
        self,
        gen_structures: list[Structure],
        metrics_obj: Metrics,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        # Get reference composition features
        assert metrics_obj._reference_composition_features is not None
        ref_composition_features: torch.Tensor = (
            metrics_obj._reference_composition_features
        )

        # Get generated composition features
        gen_features = featurize(gen_structures)
        gen_composition_features = gen_features["composition_features"].to(device)
        ref_composition_features = ref_composition_features.to(device)
        if len(ref_composition_features) > 50000:  # Subsample for efficiency
            indices = torch.randperm(len(ref_composition_features))[:50000]
            ref_composition_features = ref_composition_features[indices]

        # Compute MMD reward
        r_composition_diversity = mmd_reward(
            z_gen=gen_composition_features, z_ref=ref_composition_features
        )["r_indiv"]
        return r_composition_diversity


###############################################################################
#                                  Utils                                      #
###############################################################################


def standardize(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if x.std() < eps:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std() + eps)


def normalize(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if x.max() - x.min() < eps:
        return torch.zeros_like(x)
    x = (x - x.min()) / (x.max() - x.min() + eps)
    return x.clamp(0.0, 1.0)


def mmd_reward(z_gen, z_ref):
    """Training Diffusion Models Towards Diverse Image Generation with Reinforcement Learning."""

    def poly_k(z, y, deg=3):
        d = z.size(-1)
        return (z @ y.T / d + 1) ** deg

    M, N = len(z_gen), len(z_ref)

    k_gg = poly_k(z_gen, z_gen)
    k_rr = poly_k(z_ref, z_ref)
    k_gr = poly_k(z_gen, z_ref)

    # Compute MMD
    R_term = (k_rr.sum() - k_rr.trace()) / (N * (N - 1))
    G = k_gg.sum() - k_gg.trace()
    C = k_gr.sum()
    mmd_full = G / (M * (M - 1)) + R_term - 2 * C / (M * N)  # Eq. (11)

    # Compute individual MMD
    S = k_gg.sum(dim=1) - k_gg.diagonal()  # S_m
    T = k_gr.sum(dim=1)  # T_m

    Mp = M - 1
    Ap = Mp * (Mp - 1)
    mmd_drop = (G - 2 * S) / Ap + R_term - 2 * (C - T) / (Mp * N)

    r_indiv = mmd_drop - mmd_full
    return {"r": -mmd_full, "r_indiv": r_indiv}
