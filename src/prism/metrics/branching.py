import math
from typing import Dict, Tuple

from prism.reconstruction.protocols import PredictiveStateModel


def mean_branching_entropy(model: PredictiveStateModel, log_base: float = math.e) -> float:
    """
    Mean conditional entropy of next-state distribution per (state, symbol):

        H(S_{t+1} | S_t=s, X_{t+1}=sym) = - sum_{s'} p(s'|s,sym) log p(s'|s,sym)

    Returns the unweighted mean over all (s, sym) keys present in model.transitions.

    log_base:
      - math.e -> nats
      - 2.0    -> bits
    """
    if not model.transitions:
        return 0.0

    entropies = []
    for (_, _), dist in model.transitions.items():
        if not dist:
            continue
        h = 0.0
        for p in dist.values():
            if p <= 0.0:
                continue
            h -= p * math.log(p)
        if log_base != math.e:
            h /= math.log(log_base)
        entropies.append(h)

    return sum(entropies) / (len(entropies) or 1)


def mean_branching_entropy_weighted(model: PredictiveStateModel, log_base: float = math.e) -> float:
    """
    Weighted variant using the same proxy weights as unifilarity_score:

      w(s,1) = pi[s] * p_next_one[s]
      w(s,0) = pi[s] * (1 - p_next_one[s])

    This approximates the expected branching entropy under the model's empirical state occupancy.
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
        h = 0.0
        for p in dist.values():
            if p <= 0.0:
                continue
            h -= p * math.log(p)
        if log_base != math.e:
            h /= math.log(log_base)

        w = weight(s, sym)
        total_w += w
        total += w * h

    if total_w <= 0.0:
        return mean_branching_entropy(model, log_base=log_base)

    return total / total_w