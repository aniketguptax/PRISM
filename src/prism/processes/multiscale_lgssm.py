"""Multiscale LGSSM with predictive low-variance structure."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from prism.types import LatentState, Obs

from .protocols import Process, Sample


def _as_obs_rows(matrix: np.ndarray) -> list[Obs]:
    return [tuple(float(value) for value in row) for row in matrix]


def _as_latent_rows(matrix: np.ndarray) -> list[LatentState]:
    return [tuple(float(value) for value in row) for row in matrix]


@dataclass(frozen=True)
class MultiscaleLGSSM(Process):
    """Five-dimensional LGSSM for multiscale recovery tests.

    Coordinates: slow AR(1), damped oscillator, and two high-variance
    low-memory distractors. A random orthonormal observation map mixes all
    coordinates.
    """

    slow_a: float = 0.992
    oscillator_radius: float = 0.95
    oscillator_angle: float = math.pi / 10.0
    distractor_a1: float = 0.05
    distractor_a2: float = -0.03
    slow_process_std: float = 0.08
    oscillator_process_std: float = 0.10
    distractor_process_std: float = 1.0
    slow_loading: float = 0.80
    oscillator_loading: float = 1.60
    distractor_loading: float = 3.00
    obs_std: float = 0.15
    obs_dim: int = 8

    @property
    def name(self) -> str:
        return "multiscale_lgssm"

    @property
    def latent_dim(self) -> int:
        return 5

    def __post_init__(self) -> None:
        if self.obs_dim < self.latent_dim:
            raise ValueError("obs_dim must be >= 5.")
        if not (0.0 <= abs(self.slow_a) < 1.0):
            raise ValueError("|slow_a| must be < 1.")
        if not (0.0 <= abs(self.oscillator_radius) < 1.0):
            raise ValueError("|oscillator_radius| must be < 1.")
        if not (0.0 <= abs(self.distractor_a1) < 1.0):
            raise ValueError("|distractor_a1| must be < 1.")
        if not (0.0 <= abs(self.distractor_a2) < 1.0):
            raise ValueError("|distractor_a2| must be < 1.")
        scales = (
            self.slow_process_std,
            self.oscillator_process_std,
            self.distractor_process_std,
            self.slow_loading,
            self.oscillator_loading,
            self.distractor_loading,
            self.obs_std,
        )
        if any(scale <= 0.0 for scale in scales):
            raise ValueError("process, loading, and observation scales must be positive.")

    def transition_matrix(self) -> np.ndarray:
        theta = float(self.oscillator_angle)
        radius = float(self.oscillator_radius)
        rotation = radius * np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ],
            dtype=float,
        )
        transition = np.zeros((self.latent_dim, self.latent_dim), dtype=float)
        transition[0, 0] = float(self.slow_a)
        transition[1:3, 1:3] = rotation
        transition[3, 3] = float(self.distractor_a1)
        transition[4, 4] = float(self.distractor_a2)
        return transition

    def process_stds(self) -> np.ndarray:
        return np.asarray(
            [
                self.slow_process_std,
                self.oscillator_process_std,
                self.oscillator_process_std,
                self.distractor_process_std,
                self.distractor_process_std,
            ],
            dtype=float,
        )

    def _mixing(self, rng: np.random.Generator) -> np.ndarray:
        raw = rng.normal(size=(self.obs_dim, self.obs_dim))
        q, r = np.linalg.qr(raw)
        signs = np.sign(np.diag(r))
        signs[signs == 0.0] = 1.0
        q = q * signs
        loadings = np.diag(
            [
                self.slow_loading,
                self.oscillator_loading,
                self.oscillator_loading,
                self.distractor_loading,
                self.distractor_loading,
            ]
        )
        return q[:, : self.latent_dim] @ loadings

    def sample(self, length: int, seed: int) -> Sample:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")

        rng = np.random.default_rng(seed)
        transition = self.transition_matrix()
        process_stds = self.process_stds()
        mixing = self._mixing(rng)

        latent = np.zeros((length, self.latent_dim), dtype=float)
        obs = np.zeros((length, self.obs_dim), dtype=float)

        stationary_stds = process_stds / np.sqrt(
            np.maximum(1.0 - np.diag(transition) ** 2, 1e-9)
        )
        latent[0] = rng.normal(0.0, stationary_stds)
        obs[0] = mixing @ latent[0] + rng.normal(0.0, self.obs_std, self.obs_dim)
        for idx in range(1, length):
            latent[idx] = transition @ latent[idx - 1] + rng.normal(0.0, process_stds)
            obs[idx] = mixing @ latent[idx] + rng.normal(0.0, self.obs_std, self.obs_dim)

        return Sample(x=_as_obs_rows(obs), latent=_as_latent_rows(latent))

    def regime_labels(
        self,
        latent: np.ndarray,
        *,
        slow_bins: int = 3,
        phase_bins: int = 4,
    ) -> dict[str, np.ndarray]:
        if slow_bins < 2 or phase_bins < 2:
            raise ValueError("slow_bins and phase_bins must both be >= 2.")
        latent = np.asarray(latent, dtype=float)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent shape (T, {self.latent_dim}), got {latent.shape}.")

        slow_edges = np.quantile(latent[:, 0], np.linspace(0.0, 1.0, slow_bins + 1)[1:-1])
        slow = np.digitize(latent[:, 0], slow_edges, right=False).astype(int)

        phase_angle = np.arctan2(latent[:, 2], latent[:, 1])
        phase = np.floor(((phase_angle + math.pi) / (2.0 * math.pi)) * phase_bins).astype(int)
        phase = np.clip(phase, 0, phase_bins - 1)

        return {
            "slow": slow,
            "phase": phase,
            "joint": slow * phase_bins + phase,
        }
