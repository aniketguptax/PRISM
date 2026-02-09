from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .protocols import Obs, Representation


@dataclass(frozen=True)
class LastK(Representation):
    k: int

    def __post_init__(self):
        if self.k < 1:
            raise ValueError("k must be at least 1")

    @property
    def name(self) -> str:
        return f"last_{self.k}"

    @property
    def lookback(self) -> int:
        return self.k - 1

    def __call__(self, x: Sequence[Obs], t: int) -> Tuple[int, ...]:
        start = t - self.k + 1
        return tuple(int(v) for v in x[start : t + 1])


@dataclass(frozen=True)
class LastKWithNoise(Representation):
    k: int
    noise: Sequence[int]

    def __post_init__(self):
        if self.k < 1:
            raise ValueError("k must be at least 1")

    @property
    def name(self) -> str:
        return f"last_{self.k}_noisy"

    @property
    def lookback(self) -> int:
        return self.k - 1

    def __call__(self, x: Sequence[Obs], t: int) -> Tuple[int, ...]:
        start = t - self.k + 1
        return tuple(int(v) for v in x[start : t + 1]) + (int(self.noise[t]),)