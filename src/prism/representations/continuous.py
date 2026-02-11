from dataclasses import dataclass
from typing import Sequence

from prism.types import Obs

from .protocols import Representation


@dataclass(frozen=True)
class ISSDim(Representation):
    d: int
    dv: int = 1

    def __post_init__(self) -> None:
        if self.d < 1:
            raise ValueError("ISSDim requires d >= 1.")
        if self.dv < 1:
            raise ValueError("ISSDim requires dv >= 1.")

    @property
    def name(self) -> str:
        if self.dv == 1:
            return f"iss_d{self.d}"
        return f"iss_d{self.d}_dv{self.dv}"

    @property
    def lookback(self) -> int:
        return 0

    def __call__(self, x: Sequence[Obs], t: int) -> tuple[int, int]:
        if t < 0 or t >= len(x):
            raise IndexError(f"t must be in [0, {len(x) - 1}] for {self.name}; got {t}.")
        return (self.d, self.dv)
