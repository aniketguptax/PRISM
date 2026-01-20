import random
from dataclasses import dataclass
from typing import Optional

from .protocols import Process, Sample


@dataclass(frozen=True)
class NoisyObservation(Process):
    base: Process
    flip_p: float = 0.0  # flip observed bit with probability flip_p

    @property
    def name(self) -> str:
        return f"{self.base.name}_flip{self.flip_p}"

    def __post_init__(self):
        if not (0.0 <= self.flip_p <= 1.0):
            raise ValueError("flip_p must be between 0 and 1")

    def sample(self, length: int, seed: int) -> Sample:
        s = self.base.sample(length=length, seed=seed)
        rng = random.Random(seed + 10_000_019)  # deterministic but separated
        x = [(xi ^ 1) if rng.random() < self.flip_p else xi for xi in s.x]
        return Sample(x=x, latent=s.latent)


@dataclass(frozen=True)
class Subsample(Process):
    base: Process
    step: int = 2

    @property
    def name(self) -> str:
        return f"{self.base.name}_sub{self.step}"

    def __post_init__(self):
        if self.step < 1:
            raise ValueError("step must be >= 1")

    def sample(self, length: int, seed: int) -> Sample:
        s = self.base.sample(length=length, seed=seed)
        x = s.x[:: self.step]
        latent: Optional[list[int]] = s.latent[:: self.step] if s.latent is not None else None
        return Sample(x=x, latent=latent)