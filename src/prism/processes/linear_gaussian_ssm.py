"""Synthetic linear-Gaussian state-space process for continuous PRISM runs."""

from dataclasses import dataclass

import numpy as np

from .protocols import Process, Sample


@dataclass(frozen=True)
class LinearGaussianSSM(Process):
    """Scalar latent AR(1) with Gaussian observation noise."""

    a: float = 0.92
    c: float = 1.0
    process_std: float = 0.35
    obs_std: float = 0.25
    init_std: float = 1.0

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

    def sample(self, length: int, seed: int) -> Sample:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")
        rng = np.random.default_rng(seed)
        latent: list[float] = [float(rng.normal(0.0, self.init_std))]
        obs: list[float] = [float(self.c * latent[0] + rng.normal(0.0, self.obs_std))]
        for _ in range(1, length):
            z_next = float(self.a * latent[-1] + rng.normal(0.0, self.process_std))
            latent.append(z_next)
            obs.append(float(self.c * z_next + rng.normal(0.0, self.obs_std)))
        return Sample(x=obs, latent=latent)
