"""Hierarchical hidden process with an explicit predictive-state scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prism.types import LatentState, Obs

from .protocols import Process, Sample


def _as_latent_rows(coarse: np.ndarray, fine: np.ndarray) -> list[LatentState]:
    return [
        (int(c), int(f), int(c) * 10 + int(f))
        for c, f in zip(coarse.tolist(), fine.tolist())
    ]


@dataclass(frozen=True)
class HierarchicalPredictiveHMM(Process):
    """Discrete HMM whose hidden state has known coarse and fine predictive scales.

    The coarse state is slow and controls the fine-state dynamics. Instantaneous
    emissions are shared across coarse states, so the coarse state is not a
    simple variance or symbol-frequency label. It is instead revealed by how the
    recent observed fine phase predicts the future.
    """

    n_coarse: int = 3
    n_fine: int = 4
    alphabet_size: int = 6
    coarse_stay: float = 0.94
    fine_cycle_prob: float = 0.88
    emission_noise: float = 0.08

    @property
    def name(self) -> str:
        return "hierarchical_predictive_hmm"

    def __post_init__(self) -> None:
        if self.n_coarse != 3:
            raise ValueError("HierarchicalPredictiveHMM currently expects n_coarse=3.")
        if self.n_fine != 4:
            raise ValueError("HierarchicalPredictiveHMM currently expects n_fine=4.")
        if self.alphabet_size != 6:
            raise ValueError("HierarchicalPredictiveHMM currently expects alphabet_size=6.")
        if not (0.0 <= self.coarse_stay <= 1.0):
            raise ValueError("coarse_stay must lie in [0, 1].")
        if not (0.0 <= self.fine_cycle_prob <= 1.0):
            raise ValueError("fine_cycle_prob must lie in [0, 1].")
        if not (0.0 <= self.emission_noise <= 1.0):
            raise ValueError("emission_noise must lie in [0, 1].")

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

    def _emission_table(self) -> np.ndarray:
        table = np.zeros((self.n_coarse, self.n_fine, self.alphabet_size), dtype=float)
        uniform = np.full(self.alphabet_size, 1.0 / self.alphabet_size)
        base = np.full((self.n_fine, self.alphabet_size), 0.02, dtype=float)
        for fine in range(self.n_fine):
            base[fine, fine] = 0.76
            base[fine, 4 + (fine % 2)] = 0.12
        base = base / base.sum(axis=1, keepdims=True)
        for coarse in range(self.n_coarse):
            for fine in range(self.n_fine):
                table[coarse, fine] = (
                    (1.0 - self.emission_noise) * base[fine] + self.emission_noise * uniform
                )
        return table

    def sample(self, length: int, seed: int) -> Sample:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")

        rng = np.random.default_rng(seed)
        coarse_tpm = self._coarse_transition()
        emissions = self._emission_table()

        coarse = np.zeros(length, dtype=int)
        fine = np.zeros(length, dtype=int)
        obs = np.zeros(length, dtype=int)

        coarse[0] = int(rng.integers(0, self.n_coarse))
        fine[0] = int(rng.integers(0, self.n_fine))
        obs[0] = int(rng.choice(self.alphabet_size, p=emissions[coarse[0], fine[0]]))

        for idx in range(1, length):
            prev_coarse = coarse[idx - 1]
            prev_fine = fine[idx - 1]
            coarse[idx] = int(rng.choice(self.n_coarse, p=coarse_tpm[prev_coarse]))
            if coarse[idx] != prev_coarse:
                fine[idx] = int(rng.integers(0, self.n_fine))
            elif rng.random() < self.fine_cycle_prob:
                step = (1, -1, 2)[int(coarse[idx])]
                fine[idx] = (prev_fine + step) % self.n_fine
            else:
                fine[idx] = int(rng.integers(0, self.n_fine))
            obs[idx] = int(rng.choice(self.alphabet_size, p=emissions[coarse[idx], fine[idx]]))

        return Sample(x=[int(value) for value in obs.tolist()], latent=_as_latent_rows(coarse, fine))

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
