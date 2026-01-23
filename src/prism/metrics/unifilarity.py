import math
from typing import Dict, Tuple

from prism.reconstruction.protocols import PredictiveStateModel


def unifilarity_score(model: PredictiveStateModel) -> float:
    """
    Unifilarity score: weighted mean of local unifilarities per (state, symbol), where weights
    are proportional to empirical visitation counts of (state, symbol) pairs in training data.
    
    Local unifilarity for (s, sym) is defined as the maximum next-state probability:
        u(s, sym) = max_{s'} P(s_{t+1}=s' | s_t=s, x_{t+1}=sym)
    The overall score is:
        U = sum_{(s,sym)} w(s,sym) * u(s,sym) / sum_{(s,sym)} w(s,sym)
    where weights are:
        w(s,sym) = count((s_t=s, x_{t+1}=sym)) in training data
    """
    if not model.transitions:
        return 0.0

    total_w = 0.0
    total = 0.0

    for (s, sym), dist in model.transitions.items():
        if not dist:
            continue

        # local unifilarity: maximum next-state probability
        u = max(dist.values())

        w = float(model.sa_counts.get((s, sym), 0))
        total_w += w
        total += w * u

    if total_w <= 0.0:
        # Fallback to unweighted mean if weights are degenerate
        vals = [max(dist.values()) for dist in model.transitions.values() if dist]
        return sum(vals) / (len(vals) or 1)
     
    return max(0.0, min(1.0, total / total_w))