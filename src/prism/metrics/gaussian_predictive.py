import numpy as np
from typing import Sequence

from prism.processes.protocols import Obs
from prism.reconstruction.iss_merge import GaussianPredictiveStateModel
from prism.continuous.iss import one_step_predictive_y, _gaussian_nll

def gaussian_log_loss(x: Sequence[Obs], model: GaussianPredictiveStateModel) -> float:
    """
    Held-out one-step negative log-likelihood under macrostate Gaussian predictors.

    Assignment uses the ISS predictor (mu_t,S_t) computed from the ISS fitted on training.
    Prediction uses macrostate parameters (mu_m,Sigma_m).
    """
    y = np.asarray([float(v) for v in x], dtype=float).reshape(-1, 1)
    if y.shape[0] < 5:
        return 0.0

    mu_y, S_y, innov = one_step_predictive_y(y, model.iss)

    # we predict y_{t+1} using time t's predictor => use (t+1) index as in reconstruction
    losses = []
    for t in range(0, y.shape[0] - 1):
        mu_pred = mu_y[t + 1]
        S_pred = S_y[t + 1]
        sid = model.assign_state(mu_pred, S_pred)
        if sid is None:
            # conservative backoff: use the ISS prediction itself (still Gaussian)
            losses.append(_gaussian_nll(y[t + 1], mu_pred, S_pred))
        else:
            losses.append(_gaussian_nll(y[t + 1], model.mu[sid], model.Sigma[sid]))

    return float(sum(losses) / (len(losses) or 1))