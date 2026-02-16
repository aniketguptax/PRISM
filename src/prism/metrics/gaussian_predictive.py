"""Predictive metrics for continuous Gaussian models."""

from typing import Optional, Sequence

from prism.reconstruction.kalman_iss import GaussianPredictiveStateModel
from prism.types import Obs


def gaussian_log_loss(
    x: Sequence[Obs],
    model: GaussianPredictiveStateModel,
    *,
    context: Optional[Sequence[Obs]] = None,
) -> float:
    """Average one-step Gaussian negative log-likelihood on `x`."""
    return model.average_negative_log_likelihood(x, context=context)
