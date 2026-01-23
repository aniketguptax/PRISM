import math

from prism.metrics.unifilarity import unifilarity_score
from prism.metrics.branching import mean_branching_entropy
from prism.processes import MarkovOrder1
from prism.reconstruction import OneStepGreedyMerge
from prism.representations import LastK


def test_unifilarity_one_implies_zero_branch_entropy():
    """
    If every (state, symbol) transition is deterministic (unifilar),
    then branching entropy should be exactly 0 (up to numerical tolerance).
    """
    process = MarkovOrder1(p01=0.8, p11=0.2)
    rep = LastK(k=2)  # sufficient for Markov(1)
    recon = OneStepGreedyMerge(eps=0.0)  # keep clusters tight to avoid artefacts

    x = process.sample(length=80_000, seed=0).x
    x_train = x[: int(0.8 * len(x))]
    model = recon.fit(x_train, rep, seed=0)

    u = unifilarity_score(model)
    h = mean_branching_entropy(model, log_base=2.0)

    assert 0.0 <= u <= 1.0
    assert u > 0.999, f"Expected ~unifilar transitions, got u={u}"
    assert math.isfinite(h)
    assert h < 1e-12, f"Expected ~0 branching entropy for unifilar transitions, got h={h}"