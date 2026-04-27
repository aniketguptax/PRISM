"""Continuous hierarchical switching process for Kalman-ISS validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prism.types import LatentState, Obs

from .protocols import Process, Sample


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _as_obs_rows(matrix: np.ndarray) -> list[Obs]:
    return [tuple(float(value) for value in row) for row in matrix]


def _as_latent_rows(coarse: np.ndarray, fine: np.ndarray) -> list[LatentState]:
    return [
        (int(c), int(f), int(c) * 10 + int(f))
        for c, f in zip(coarse.tolist(), fine.tolist())
    ]


@dataclass(frozen=True)
class HierarchicalSwitchingGaussian(Process):
    n_coarse: int = 3
    n_fine: int = 4
    obs_dim: int = 8
    coarse_stay: float = 0.96
    fine_cycle_prob: float = 0.90
    emission_std: float = 0.20
    embedding_scale: float = 1.0

    @property
    def name(self) -> str:
        return "hierarchical_switching_gaussian"

    def __post_init__(self) -> None:
        if self.n_coarse != 3:
            raise ValueError("HierarchicalSwitchingGaussian currently expects n_coarse=3.")
        if self.n_fine != 4:
            raise ValueError("HierarchicalSwitchingGaussian currently expects n_fine=4.")
        if self.obs_dim < 4:
            raise ValueError("obs_dim must be at least 4.")
        if not (0.0 <= self.coarse_stay <= 1.0):
            raise ValueError("coarse_stay must lie in [0, 1].")
        if not (0.0 <= self.fine_cycle_prob <= 1.0):
            raise ValueError("fine_cycle_prob must lie in [0, 1].")
        if self.emission_std <= 0.0:
            raise ValueError("emission_std must be positive.")
        if self.embedding_scale <= 0.0:
            raise ValueError("embedding_scale must be positive.")

    def _coarse_transition(self) -> np.ndarray:
        move = (1.0 - self.coarse_stay) / 2.0
        return np.array(
            [
                [self.coarse_stay, move, move],
                [move, self.coarse_stay, move],
                [move, move, self.coarse_stay],
            ],
            dtype=float,
        )

    def _fine_embeddings(self) -> np.ndarray:
        angles = 2.0 * np.pi * np.arange(self.n_fine, dtype=float) / self.n_fine
        return self.embedding_scale * np.column_stack(
            [
                np.cos(angles),
                np.sin(angles),
                0.5 * np.cos(2.0 * angles),
                0.5 * np.sin(2.0 * angles),
            ]
        )

    def _mixing(self, rng: np.random.Generator) -> np.ndarray:
        raw = rng.normal(size=(self.obs_dim, self.obs_dim))
        q, r = np.linalg.qr(raw)
        signs = np.sign(np.diag(r))
        signs[signs == 0.0] = 1.0
        q = q * signs
        return _normalise_rows(q[:, :4])

    def sample(self, length: int, seed: int) -> Sample:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")

        rng = np.random.default_rng(seed)
        coarse_tpm = self._coarse_transition()
        embeddings = self._fine_embeddings()
        mixing = self._mixing(rng)

        coarse = np.zeros(length, dtype=int)
        fine = np.zeros(length, dtype=int)
        obs = np.zeros((length, self.obs_dim), dtype=float)

        coarse[0] = int(rng.integers(0, self.n_coarse))
        fine[0] = int(rng.integers(0, self.n_fine))
        obs[0] = mixing @ embeddings[fine[0]] + rng.normal(0.0, self.emission_std, self.obs_dim)

        steps = (1, -1, 2)
        for idx in range(1, length):
            prev_coarse = coarse[idx - 1]
            prev_fine = fine[idx - 1]
            coarse[idx] = int(rng.choice(self.n_coarse, p=coarse_tpm[prev_coarse]))
            if coarse[idx] != prev_coarse:
                fine[idx] = int(rng.integers(0, self.n_fine))
            elif rng.random() < self.fine_cycle_prob:
                fine[idx] = (prev_fine + steps[int(coarse[idx])]) % self.n_fine
            else:
                fine[idx] = int(rng.integers(0, self.n_fine))
            obs[idx] = mixing @ embeddings[fine[idx]] + rng.normal(
                0.0,
                self.emission_std,
                self.obs_dim,
            )

        return Sample(x=_as_obs_rows(obs), latent=_as_latent_rows(coarse, fine))

    def regime_labels(self, latent: np.ndarray) -> dict[str, np.ndarray]:
        latent = np.asarray(latent, dtype=int)
        if latent.ndim != 2 or latent.shape[1] < 2:
            raise ValueError(f"Expected latent shape (T, >=2), got {latent.shape}.")
        coarse = latent[:, 0].astype(int)
        fine = latent[:, 1].astype(int)
        return {
            "coarse": coarse,
            "fine": fine,
            "joint": coarse * self.n_fine + fine,
        }
