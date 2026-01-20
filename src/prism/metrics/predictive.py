import math
from typing import List

from prism.reconstruction.protocols import PredictiveStateModel


def log_loss(
    x: List[int],
    rep,
    model: PredictiveStateModel,
    backoff_p: float = 0.5,
) -> float:
    min_t = rep.lookback
    eps = 1e-12
    losses = []

    for t in range(min_t, len(x) - 1):
        r = rep(x, t)
        s = model.rep_to_state.get(r)
        
        if s is None:
            p1 = backoff_p
        else:
            p1 = model.p_next_one.get(s, backoff_p)
        
        y = x[t + 1]
        p = p1 if y == 1 else (1.0 - p1)
        losses.append(-math.log(max(p, eps)))

    return sum(losses) / (len(losses) or 1)