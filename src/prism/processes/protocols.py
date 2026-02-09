from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Union

Obs = Union[int, float]

@dataclass(frozen=True)
class Sample:
    x: Sequence[Obs]
    latent: Optional[Sequence[int]] = None


class Process(Protocol):
    @property
    def name(self) -> str:
        ...

    def sample(self, length: int, seed: int) -> Sample:
        ...