from dataclasses import dataclass
from typing import Dict, Hashable, Mapping, Protocol, Sequence, Tuple, TypeVar

from prism.representations.protocols import Representation
from prism.types import Obs

Rep = Hashable


@dataclass(frozen=True)
class PredictiveStateModel:
    rep_to_state: Dict[Rep, int]
    p_next_one: Dict[int, float]
    pi: Dict[int, float]
    transitions: Dict[Tuple[int, int], Dict[int, float]]
    sa_counts: Dict[Tuple[int, int], int]
    valid: bool = True
    invalid_reason: str = ""


class TransitionModel(Protocol):
    @property
    def transitions(self) -> Mapping[Tuple[int, int], Mapping[int, float]]:
        ...

    @property
    def sa_counts(self) -> Mapping[Tuple[int, int], int]:
        ...


class OccupancyModel(Protocol):
    @property
    def pi(self) -> Mapping[int, float]:
        ...


ModelT = TypeVar("ModelT", covariant=True)


class Reconstructor(Protocol[ModelT]):
    @property
    def name(self) -> str:
        ...

    def fit(self, x_train: Sequence[Obs], rep: Representation, seed: int = 0) -> ModelT:
        ...
