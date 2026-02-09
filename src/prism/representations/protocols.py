from typing import Hashable, Protocol, Sequence, Union

Obs = Union[int, float]
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