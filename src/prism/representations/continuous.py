from dataclasses import dataclass
from typing import Sequence

from .protocols import Representation, Obs

@dataclass(frozen=True)
class ISSDim(Representation):
    d: int

    @property
    def name(self) -> str:
        return f"iss_d{self.d}"

    @property
    def lookback(self) -> int:
        return 0

    def __call__(self, x: Sequence[Obs], t: int):
        # unused; reconstructor uses only x_train and d encoded in this rep
        return None