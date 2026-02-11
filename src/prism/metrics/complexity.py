import math
from typing import Union

from prism.reconstruction.kalman_iss import GaussianPredictiveStateModel
from prism.reconstruction.protocols import PredictiveStateModel


def statistical_complexity(model: Union[PredictiveStateModel, GaussianPredictiveStateModel]) -> float:
    eps = 1e-18
    return -sum(p * math.log(max(p, eps)) for p in model.pi.values())


def n_states(model: Union[PredictiveStateModel, GaussianPredictiveStateModel]) -> int:
    if isinstance(model, GaussianPredictiveStateModel):
        return int(model.n_macro_states)
    return len(set(model.rep_to_state.values()))
