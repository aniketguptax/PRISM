from typing import Hashable, Protocol, Sequence

from prism.types import Obs

Rep = Hashable


class Representation(Protocol):
    @property
    def name(self) -> str:
        ...
        
    @property
    def lookback(self) -> int:
        ...

    def __call__(self, x: Sequence[Obs], t: int) -> Rep:
        ...
