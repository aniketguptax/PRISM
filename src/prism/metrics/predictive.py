"""Predictive metrics for discrete binary models."""

import math
from typing import Sequence

from prism.reconstruction.protocols import PredictiveStateModel
from prism.representations.protocols import Representation
from prism.types import Obs


def _binary_symbol(value: Obs, *, index: int) -> int:
    if isinstance(value, bool):
        out = int(value)
    elif isinstance(value, int):
        out = value
    else:
        raise TypeError(
            f"log_loss expects binary integer observations, got {type(value).__name__} at index {index}."
        )
    if out not in (0, 1):
        raise ValueError(f"log_loss expects observations in {{0,1}}, got {out} at index {index}.")
    return out


def log_loss(
    x: Sequence[Obs],
    rep: Representation,
    model: PredictiveStateModel,
    backoff_p: float = 0.5,
) -> float:
    """Average one-step Bernoulli negative log-likelihood."""
    min_t = rep.lookback
    eps = 1e-12
    losses: list[float] = []
    for t in range(min_t, len(x) - 1):
        r = rep(x, t)
        s = model.rep_to_state.get(r)
        p1 = model.p_next_one[s] if s is not None else backoff_p
        y = _binary_symbol(x[t + 1], index=t + 1)
        p = p1 if y == 1 else (1.0 - p1)
        losses.append(-math.log(max(p, eps)))
    return sum(losses) / (len(losses) or 1)
