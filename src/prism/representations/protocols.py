from typing import Hashable, List, Protocol


Rep = Hashable


class Representation(Protocol):
    @property
    def name(self) -> str:
        ...
        
    @property
    def lookback(self) -> int:
        ...

    def __call__(self, x: List[int], t: int) -> Rep:
        ...