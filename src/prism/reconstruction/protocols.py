from dataclasses import dataclass
from typing import Dict, Hashable, Protocol, Sequence, Tuple

from prism.representations.protocols import Obs


Rep = Hashable


@dataclass
class PredictiveStateModel:
    # Mapping from representation value r to macrostate s
    rep_to_state: Dict[Rep, int]
    
    # One-step predictive parameter: P(x_{t+1}=1 | state=s)
    p_next_one: Dict[int, float]
    
    # Empirical state occupancy (over visited states in training)
    pi: Dict[int, float]
    
    # Stochastic transition model:
    # P(s_{t+1} = sp | s_t = s, x_{t+1} = sym)
    # keyed by (s, sym), valued by dict of {sp: prob}
    transitions: Dict[Tuple[int, int], Dict[int, float]]
    
    # Empirical counts of (state, sym) occurences in training
    sa_counts: Dict[Tuple[int, int], int]


class Reconstructor(Protocol):
    @property
    def name(self) -> str:
        ...

    def fit(self, x_train: Sequence[Obs], rep, seed: int = 0) -> PredictiveStateModel:
        ...