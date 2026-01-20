import math
from typing import Dict, Tuple

from prism.reconstruction.protocols import PredictiveStateModel


def unifilarity_score(model: PredictiveStateModel) -> float:
    """
    Weighted unifilarity score in [0, 1].

    For each (state s, symbol a), define u(s,a) = max_{s'} P(s'|s,a).
    Aggregate as a weighted average with weights proportional to how often (s,a)
    occurs in the model's empirical state distribution pi and the emission model
    p_next_one.

    Weight approximation:
      w(s,1) = pi[s] * p_next_one[s]
      w(s,0) = pi[s] * (1 - p_next_one[s])

    This avoids needing raw transition counts and is stable across runs.
    """
    if not model.transitions:
        return 0.0

    def weight(s: int, sym: int) -> float:
        pi_s = model.pi.get(s, 0.0)
        p1 = model.p_next_one.get(s, 0.5)
        return pi_s * (p1 if sym == 1 else (1.0 - p1))

    total_w = 0.0
    total = 0.0

    for (s, sym), dist in model.transitions.items():
        if not dist:
            continue

        # local unifilarity: maximum next-state probability
        u = max(dist.values())

        w = weight(s, sym)
        total_w += w
        total += w * u

    if total_w <= 0.0:
        # Fallback to unweighted mean if weights are degenerate
        vals = [max(dist.values()) for dist in model.transitions.values() if dist]
        return sum(vals) / (len(vals) or 1)

    score = total / total_w
    # numerical guard
    return max(0.0, min(1.0, score))