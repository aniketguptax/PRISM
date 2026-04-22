"""Block-modular linear Gaussian state-space generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from prism.types import LatentState, Obs

from .protocols import Process, Sample


def _slow_real_block(eigvals: tuple[float, float]) -> np.ndarray:
    return np.diag(np.asarray(eigvals, dtype=float))


def _rotational_block(modulus: float, angle_rad: float) -> np.ndarray:
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return modulus * np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def _haar_orthonormal(rng: np.random.Generator, n_rows: int, n_cols: int) -> np.ndarray:
    if n_rows < n_cols:
        raise ValueError(f"Haar mixing requires p >= d, got p={n_rows}, d={n_cols}.")
    raw = rng.normal(size=(n_rows, n_rows))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs
    return q[:, :n_cols]


def _aligned_mixing(
    rng: np.random.Generator,
    n_rows: int,
    blocks: Sequence[int],
) -> np.ndarray:
    if sum(blocks) <= 0:
        raise ValueError("blocks must contain at least one positive entry.")

    base_rows = n_rows // len(blocks)
    remainder = n_rows - base_rows * len(blocks)
    rows_per_block = [
        base_rows + (1 if idx < remainder else 0)
        for idx in range(len(blocks))
    ]
    if any(count == 0 for count in rows_per_block):
        raise ValueError(f"aligned mixing needs p >= number of blocks, got p={n_rows}.")

    matrix = np.zeros((n_rows, sum(blocks)), dtype=float)
    row_start = 0
    col_start = 0
    for block_dim, block_rows in zip(blocks, rows_per_block):
        sample = rng.normal(size=(block_rows, block_dim))
        if block_rows >= block_dim:
            q, _ = np.linalg.qr(sample)
            block = q[:, :block_dim]
        else:
            block = sample
        matrix[row_start : row_start + block_rows, col_start : col_start + block_dim] = block
        row_start += block_rows
        col_start += block_dim
    return matrix


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _as_obs_rows(matrix: np.ndarray) -> list[Obs]:
    if matrix.shape[1] == 1:
        return [float(value) for value in matrix[:, 0]]
    return [tuple(float(value) for value in row) for row in matrix]


def _as_latent_rows(matrix: np.ndarray) -> list[LatentState]:
    if matrix.shape[1] == 1:
        return [float(value) for value in matrix[:, 0]]
    return [tuple(float(value) for value in row) for row in matrix]


@dataclass(frozen=True)
class BlockModularLGSSM(Process):
    """Two-block LGSSM with controllable cross-block coupling."""

    block_dims: tuple[int, int] = (2, 2)
    slow_eigvals: tuple[float, float] = (0.92, 0.85)
    rotation_modulus: float = 0.70
    rotation_period: float = 12.0
    coupling: float = 0.05
    process_std: float = 0.35
    obs_std: float = 0.25
    obs_dim: int = 8
    obs_design: str = "random"
    init_std: float = 1.0

    @property
    def name(self) -> str:
        if self.obs_design == "aligned":
            return f"block_modular_lgssm_aligned_eps{self.coupling:g}"
        return f"block_modular_lgssm_eps{self.coupling:g}"

    def __post_init__(self) -> None:
        if len(self.block_dims) != 2:
            raise ValueError("BlockModularLGSSM expects exactly two blocks.")
        if any(block_dim != 2 for block_dim in self.block_dims):
            raise ValueError("Each latent block is currently fixed at dimension 2.")
        if max(abs(self.slow_eigvals[0]), abs(self.slow_eigvals[1])) >= 1.0:
            raise ValueError("Slow-block eigenvalues must satisfy |lambda| < 1.")
        if not (0.0 < self.rotation_modulus < 1.0):
            raise ValueError("rotation_modulus must lie in (0, 1).")
        if self.rotation_period <= 1.0:
            raise ValueError("rotation_period must be > 1.")
        if self.coupling < 0.0:
            raise ValueError("coupling must be >= 0.")
        if self.process_std <= 0.0 or self.obs_std <= 0.0 or self.init_std <= 0.0:
            raise ValueError("Noise scales must be strictly positive.")
        if self.obs_dim < sum(self.block_dims):
            raise ValueError("obs_dim must be at least the latent dimension.")
        if self.obs_design not in {"random", "aligned"}:
            raise ValueError("obs_design must be 'random' or 'aligned'.")

    def _build_A(self, rng: np.random.Generator) -> np.ndarray:
        latent_dim = sum(self.block_dims)
        transition = np.zeros((latent_dim, latent_dim), dtype=float)
        transition[:2, :2] = _slow_real_block(self.slow_eigvals)
        transition[2:, 2:] = _rotational_block(
            self.rotation_modulus,
            2.0 * np.pi / float(self.rotation_period),
        )

        if self.coupling > 0.0:
            mask = np.ones((latent_dim, latent_dim), dtype=float)
            mask[:2, :2] = 0.0
            mask[2:, 2:] = 0.0
            off_block = rng.normal(size=(latent_dim, latent_dim)) * mask
            transition = transition + (self.coupling / np.sqrt(latent_dim)) * off_block

        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
        if spectral_radius >= 0.99:
            transition = transition * (0.99 / spectral_radius)
        return transition

    def _build_C(self, rng: np.random.Generator) -> np.ndarray:
        latent_dim = sum(self.block_dims)
        if self.obs_design == "random":
            mixing = _haar_orthonormal(rng, n_rows=self.obs_dim, n_cols=latent_dim)
        else:
            mixing = _aligned_mixing(rng, n_rows=self.obs_dim, blocks=self.block_dims)
        return _normalise_rows(mixing)

    def sample(self, length: int, seed: int) -> Sample:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")

        rng = np.random.default_rng(seed)
        transition = self._build_A(rng)
        mixing = self._build_C(rng)
        latent_dim = transition.shape[0]
        obs_dim = mixing.shape[0]

        latent = np.zeros((length, latent_dim), dtype=float)
        obs = np.zeros((length, obs_dim), dtype=float)

        latent[0] = rng.normal(0.0, self.init_std, size=latent_dim)
        obs[0] = mixing @ latent[0] + rng.normal(0.0, self.obs_std, size=obs_dim)
        for idx in range(1, length):
            latent[idx] = transition @ latent[idx - 1] + rng.normal(
                0.0,
                self.process_std,
                size=latent_dim,
            )
            obs[idx] = mixing @ latent[idx] + rng.normal(0.0, self.obs_std, size=obs_dim)

        return Sample(x=_as_obs_rows(obs), latent=_as_latent_rows(latent))

    def block_attribution(self, latent: np.ndarray) -> np.ndarray:
        """Label each step by the dominant latent block."""
        latent = np.asarray(latent, dtype=float)
        if latent.ndim != 2 or latent.shape[1] != sum(self.block_dims):
            raise ValueError(
                f"Expected latent shape (T, {sum(self.block_dims)}), got {latent.shape}."
            )

        squared = latent**2
        norm_first = squared[:, : self.block_dims[0]].sum(axis=1)
        norm_second = squared[:, self.block_dims[0] :].sum(axis=1)
        return (norm_second > norm_first).astype(int)

    def regime_labels(
        self,
        latent: np.ndarray,
        *,
        slow_bins: int = 3,
        phase_bins: int = 6,
    ) -> dict[str, np.ndarray]:
        """Return slow-block, phase-block, and joint regime labels."""
        latent = np.asarray(latent, dtype=float)
        latent_dim = sum(self.block_dims)
        if latent.ndim != 2 or latent.shape[1] != latent_dim:
            raise ValueError(f"Expected latent shape (T, {latent_dim}), got {latent.shape}.")
        if slow_bins < 2 or phase_bins < 2:
            raise ValueError("slow_bins and phase_bins must be at least 2.")

        slow_signal = latent[:, 0]
        slow_edges = np.quantile(slow_signal, np.linspace(0.0, 1.0, slow_bins + 1)[1:-1])
        slow = np.digitize(slow_signal, slow_edges, right=False).astype(int)

        theta = np.arctan2(latent[:, 3], latent[:, 2])
        phase = np.floor(((theta + np.pi) / (2.0 * np.pi)) * phase_bins).astype(int)
        phase = np.clip(phase, 0, phase_bins - 1)

        joint = slow * phase_bins + phase
        return {
            "slow_block": slow,
            "phase_block": phase,
            "joint": joint,
        }
