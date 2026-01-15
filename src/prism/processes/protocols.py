from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class Sample:
    x: List[int]
    latent: Optional[List[int]] = None


class Process(Protocol):
    @property
    def name(self) -> str:
        ...

    def sample(self, length: int, seed: int) -> Sample:
        ...