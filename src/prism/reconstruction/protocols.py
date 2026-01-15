from dataclasses import dataclass
from typing import Dict, Hashable, List, Protocol, Tuple


Rep = Hashable


@dataclass
class PredictiveStateModel:
    rep_to_state: Dict[Rep, int]
    p_next_one: Dict[int, float]
    pi: Dict[int, float]
    transitions: Dict[Tuple[int, int], int]


class Reconstructor(Protocol):
    @property
    def name(self) -> str:
        ...

    def fit(self, x_train: List[int], rep, seed: int = 0) -> PredictiveStateModel:
        ...