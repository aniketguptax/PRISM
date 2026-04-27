"""LGSSM with low-variance predictive structure and high-variance distractors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prism.types import LatentState, Obs

from .protocols import Process, Sample


def _as_obs_rows(matrix: np.ndarray) -> list[Obs]:
    return [tuple(float(value) for value in row) for row in matrix]


def _as_latent_rows(matrix: np.ndarray) -> list[LatentState]:
    return [tuple(float(value) for value in row) for row in matrix]


@dataclass(frozen=True)
class PredictiveLowVarianceLGSSM(Process):
    slow_a: float = 0.985
    fast_a: float = 0.15
    slow_process_std: float = 0.10
    fast_process_std: float = 1.0
    slow_loading: float = 0.80
    fast_loading: float = 1.0
    obs_std: float = 0.25
    obs_dim: int = 8
    latent_dim: int = 3

    @property
    def name(self) -> str:
        return "predictive_low_variance_lgssm"

    def __post_init__(self) -> None:
        if self.latent_dim < 2:
            raise ValueError("latent_dim must be at least 2.")
        if self.obs_dim < self.latent_dim:
            raise ValueError("obs_dim must be >= latent_dim.")
        if not (0.0 <= abs(self.slow_a) < 1.0):
            raise ValueError("|slow_a| must be < 1.")
        if not (0.0 <= abs(self.fast_a) < 1.0):
            raise ValueError("|fast_a| must be < 1.")
        if self.slow_process_std <= 0.0 or self.fast_process_std <= 0.0 or self.obs_std <= 0.0:
            raise ValueError("noise scales must be positive.")

    def _mixing(self, rng: np.random.Generator) -> np.ndarray:
        raw = rng.normal(size=(self.obs_dim, self.obs_dim))
        q, r = np.linalg.qr(raw)
        signs = np.sign(np.diag(r))
        signs[signs == 0.0] = 1.0
        q = q * signs
        loadings = np.diag([self.slow_loading] + [self.fast_loading] * (self.latent_dim - 1))
        return q[:, : self.latent_dim] @ loadings

    def sample(self, length: int, seed: int) -> Sample:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")
        rng = np.random.default_rng(seed)
        transition = np.diag([self.slow_a] + [self.fast_a] * (self.latent_dim - 1))
        process_std = np.asarray(
            [self.slow_process_std] + [self.fast_process_std] * (self.latent_dim - 1),
            dtype=float,
        )
        mixing = self._mixing(rng)

        latent = np.zeros((length, self.latent_dim), dtype=float)
        obs = np.zeros((length, self.obs_dim), dtype=float)
        stationary_std = process_std / np.sqrt(1.0 - np.diag(transition) ** 2)
        latent[0] = rng.normal(0.0, stationary_std)
        obs[0] = mixing @ latent[0] + rng.normal(0.0, self.obs_std, self.obs_dim)
        for idx in range(1, length):
            latent[idx] = transition @ latent[idx - 1] + rng.normal(0.0, process_std)
            obs[idx] = mixing @ latent[idx] + rng.normal(0.0, self.obs_std, self.obs_dim)
        return Sample(x=_as_obs_rows(obs), latent=_as_latent_rows(latent))

    def regime_labels(
        self,
        latent: np.ndarray,
        *,
        slow_bins: int = 3,
        fast_bins: int = 3,
    ) -> dict[str, np.ndarray]:
        latent = np.asarray(latent, dtype=float)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent shape (T, {self.latent_dim}), got {latent.shape}.")
        slow_edges = np.quantile(latent[:, 0], np.linspace(0.0, 1.0, slow_bins + 1)[1:-1])
        slow = np.digitize(latent[:, 0], slow_edges, right=False).astype(int)

        fast_energy = np.linalg.norm(latent[:, 1:], axis=1)
        fast_edges = np.quantile(fast_energy, np.linspace(0.0, 1.0, fast_bins + 1)[1:-1])
        fast = np.digitize(fast_energy, fast_edges, right=False).astype(int)
        return {
            "slow": slow,
            "fast": fast,
            "joint": slow * fast_bins + fast,
        }
