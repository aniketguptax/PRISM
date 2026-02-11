from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from prism.types import LatentState, Obs
from .protocols import Process, Sample


def _as_obs_rows(matrix: np.ndarray) -> list[Obs]:
    if matrix.shape[1] == 1:
        return [float(v) for v in matrix[:, 0]]
    return [tuple(float(v) for v in row) for row in matrix]


def _as_latent_rows(matrix: np.ndarray) -> list[LatentState]:
    if matrix.shape[1] == 1:
        return [float(v) for v in matrix[:, 0]]
    return [tuple(float(v) for v in row) for row in matrix]


def _stable_matrix(rng: np.random.Generator, dim: int, a: float, coupling_std: float) -> np.ndarray:
    base = np.eye(dim, dtype=float) * a
    if dim > 1 and coupling_std > 0.0:
        noise = rng.normal(scale=coupling_std / max(dim, 1), size=(dim, dim))
        np.fill_diagonal(noise, 0.0)
        base = base + noise
    eigvals = np.linalg.eigvals(base)
    spectral_radius = float(np.max(np.abs(eigvals)))
    if spectral_radius >= 0.99:
        base *= 0.99 / spectral_radius
    return base


@dataclass(frozen=True)
class LinearGaussianSSM(Process):
    a: float = 0.92
    c: float = 1.0
    process_std: float = 0.35
    obs_std: float = 0.25
    init_std: float = 1.0
    latent_dim: int = 2
    obs_dim: int = 3
    coupling_std: float = 0.08

    @property
    def name(self) -> str:
        return "linear_gaussian_ssm"

    def __post_init__(self) -> None:
        if abs(self.a) >= 1.0:
            raise ValueError("LinearGaussianSSM requires |a| < 1 for stationarity.")
        if self.process_std <= 0.0:
            raise ValueError("process_std must be > 0.")
        if self.obs_std <= 0.0:
            raise ValueError("obs_std must be > 0.")
        if self.init_std <= 0.0:
            raise ValueError("init_std must be > 0.")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1.")
        if self.obs_dim < 1:
            raise ValueError("obs_dim must be >= 1.")
        if self.coupling_std < 0.0:
            raise ValueError("coupling_std must be >= 0.")

    def sample(self, length: int, seed: int) -> Sample:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")
        rng = np.random.default_rng(seed)
        A = _stable_matrix(rng, dim=self.latent_dim, a=self.a, coupling_std=self.coupling_std)
        C = rng.normal(scale=self.c / np.sqrt(self.latent_dim), size=(self.obs_dim, self.latent_dim))
        q_std = float(self.process_std)
        r_std = float(self.obs_std)

        latent = np.zeros((length, self.latent_dim), dtype=float)
        obs = np.zeros((length, self.obs_dim), dtype=float)
        latent[0] = rng.normal(0.0, self.init_std, size=self.latent_dim)
        obs[0] = C @ latent[0] + rng.normal(0.0, r_std, size=self.obs_dim)

        for t in range(1, length):
            latent[t] = A @ latent[t - 1] + rng.normal(0.0, q_std, size=self.latent_dim)
            obs[t] = C @ latent[t] + rng.normal(0.0, r_std, size=self.obs_dim)

        return Sample(x=_as_obs_rows(obs), latent=_as_latent_rows(latent))
