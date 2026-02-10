"""Continuous representation descriptors."""

from dataclasses import dataclass
from typing import Sequence

from prism.types import Obs

from .protocols import Representation


@dataclass(frozen=True)
class ISSDim(Representation):
    """Representation that encodes the latent state dimension for Kalman ISS."""

    d: int

    def __post_init__(self) -> None:
        if self.d < 1:
            raise ValueError("ISSDim requires d >= 1.")

    @property
    def name(self) -> str:
        return f"iss_d{self.d}"

    @property
    def lookback(self) -> int:
        return 0

    def __call__(self, x: Sequence[Obs], t: int) -> int:
        if t < 0 or t >= len(x):
            raise IndexError(f"t must be in [0, {len(x) - 1}] for {self.name}; got {t}.")
        return self.d
