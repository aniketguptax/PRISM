import math
from prism.reconstruction.protocols import TransitionModel


def mean_branching_entropy(model: TransitionModel, log_base: float = math.e) -> float:
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


def mean_branching_entropy_weighted(model: TransitionModel, log_base: float = math.e) -> float:
    if not model.transitions:
        return 0.0

    total_w = 0.0
    total = 0.0

    for (s, sym), dist in model.transitions.items():
        if not dist:
            continue

        h = 0.0
        for p in dist.values():
            if p > 0.0:
                h -= p * math.log(p)
        if log_base != math.e:
            h /= math.log(log_base)

        w = float(model.sa_counts.get((s, sym), 0))
        total_w += w
        total += w * h

    if total_w <= 0.0:
        return mean_branching_entropy(model, log_base=log_base)

    return total / total_w
