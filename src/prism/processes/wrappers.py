import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .protocols import Process, Sample, Obs


def _ensure_binary_ints(x: Sequence[Obs], *, name: str) -> List[int]:
    """
    Ensure observations are ints in {0,1}. Raise a helpful error otherwise.
    Returns a concrete List[int] for downstream wrappers.
    """
    out: List[int] = []
    for i, v in enumerate(x):
        if isinstance(v, bool):
            iv = int(v)
        elif isinstance(v, int):
            iv = v
        else:
            raise TypeError(
                f"{name} expects a binary *discrete* process (ints 0/1), but got a non-int "
                f"observation at index {i}: {v!r} (type {type(v).__name__}). "
                f"Do not apply NoisyObservation/Subsample to continuous data."
            )
        if iv not in (0, 1):
            raise ValueError(
                f"{name} expects observations in {{0,1}}, but got {iv} at index {i}."
            )
        out.append(iv)
    return out


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
        x_int = _ensure_binary_ints(s.x, name=self.name)
        
        rng = random.Random(seed + 10_000_019)  # deterministic but separated
        x = [(xi ^ 1) if rng.random() < self.flip_p else xi for xi in x_int]
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
        latent = s.latent[:: self.step] if s.latent is not None else None
        return Sample(x=x, latent=latent)