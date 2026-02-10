"""Model-size and complexity metrics."""

import math
from typing import Union

from prism.reconstruction.kalman_iss import GaussianPredictiveStateModel
from prism.reconstruction.protocols import PredictiveStateModel


def statistical_complexity(model: PredictiveStateModel) -> float:
    """Empirical statistical complexity for discrete state occupancy."""
    eps = 1e-18
    return -sum(p * math.log(max(p, eps)) for p in model.pi.values())


def n_states(model: Union[PredictiveStateModel, GaussianPredictiveStateModel]) -> int:
    """Number of states for discrete models, latent dimension for ISS models."""
    if isinstance(model, GaussianPredictiveStateModel):
        return model.latent_dim
    return len(set(model.rep_to_state.values()))
