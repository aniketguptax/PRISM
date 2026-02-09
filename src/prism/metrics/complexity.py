import math
from prism.reconstruction.protocols import PredictiveStateModel


def statistical_complexity(model: PredictiveStateModel) -> float:
    eps = 1e-18
    return -sum(p * math.log(max(p, eps)) for p in model.pi.values())


def n_states(model: PredictiveStateModel) -> int:
    if hasattr(model, "rep_to_state"):
        return len(set(model.rep_to_state.values()))
    if hasattr(model, "mu"):
        return len(getattr(model, "mu"))
    raise TypeError(f"Unsupported model type for n_states: {type(model)}")