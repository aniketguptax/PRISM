"""Reconstruction protocols and discrete predictive model types."""

from dataclasses import dataclass
from typing import Dict, Hashable, Protocol, Sequence, Tuple, TypeVar

from prism.representations.protocols import Representation
from prism.types import Obs

Rep = Hashable


@dataclass(frozen=True)
class PredictiveStateModel:
    """Discrete predictive-state model reconstructed from binary observations."""

    rep_to_state: Dict[Rep, int]
    p_next_one: Dict[int, float]
    pi: Dict[int, float]
    transitions: Dict[Tuple[int, int], Dict[int, float]]
    sa_counts: Dict[Tuple[int, int], int]


ModelT = TypeVar("ModelT", covariant=True)


class Reconstructor(Protocol[ModelT]):
    """Protocol implemented by all reconstruction backends."""

    @property
    def name(self) -> str:
        ...

    def fit(self, x_train: Sequence[Obs], rep: Representation, seed: int = 0) -> ModelT:
        ...
