"""Discrete representation functions."""

from dataclasses import dataclass
from typing import Sequence, Tuple

from prism.types import Obs

from .protocols import Representation


def _require_binary_int(value: Obs, *, index: int, name: str) -> int:
    if isinstance(value, bool):
        ivalue = int(value)
    elif isinstance(value, int):
        ivalue = value
    else:
        raise TypeError(
            f"{name} expects binary integer observations; got {type(value).__name__} at index {index}."
        )
    if ivalue not in (0, 1):
        raise ValueError(f"{name} expects observations in {{0,1}}; got {ivalue} at index {index}.")
    return ivalue


@dataclass(frozen=True)
class LastK(Representation):
    """Tuple of the last `k` binary observations ending at index `t`."""

    k: int

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")

    @property
    def name(self) -> str:
        return f"last_{self.k}"

    @property
    def lookback(self) -> int:
        return self.k - 1

    def __call__(self, x: Sequence[Obs], t: int) -> Tuple[int, ...]:
        if t < self.lookback:
            raise IndexError(f"{self.name} requires t >= {self.lookback}, got {t}.")
        start = t - self.k + 1
        return tuple(_require_binary_int(v, index=start + i, name=self.name) for i, v in enumerate(x[start : t + 1]))


@dataclass(frozen=True)
class LastKWithNoise(Representation):
    """`LastK` augmented with a deterministic binary noise bit at time `t`."""

    k: int
    noise: Sequence[int]

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")

    @property
    def name(self) -> str:
        return f"last_{self.k}_noisy"

    @property
    def lookback(self) -> int:
        return self.k - 1

    def __call__(self, x: Sequence[Obs], t: int) -> Tuple[int, ...]:
        if t >= len(self.noise):
            raise IndexError(
                f"{self.name} received t={t} but noise has length {len(self.noise)}."
            )
        if t < self.lookback:
            raise IndexError(f"{self.name} requires t >= {self.lookback}, got {t}.")
        start = t - self.k + 1
        window = tuple(_require_binary_int(v, index=start + i, name=self.name) for i, v in enumerate(x[start : t + 1]))
        noise_bit = _require_binary_int(self.noise[t], index=t, name=f"{self.name}.noise")
        return window + (noise_bit,)
