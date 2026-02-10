from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from prism.types import LatentState, Obs

@dataclass(frozen=True)
class Sample:
    x: Sequence[Obs]
    latent: Optional[Sequence[LatentState]] = None


class Process(Protocol):
    @property
    def name(self) -> str:
        ...

    def sample(self, length: int, seed: int) -> Sample:
        ...
