from prism.metrics.branching import mean_branching_entropy
from prism.processes import EvenProcess
from prism.reconstruction import OneStepGreedyMerge
from prism.representations import LastK


def test_branching_entropy_nonnegative_and_finite():
    process = EvenProcess(p_emit_one=0.7)
    rep = LastK(k=3)
    recon = OneStepGreedyMerge(eps=0.02)

    x = process.sample(length=50_000, seed=0).x
    x_train = x[: int(0.8 * len(x))]
    model = recon.fit(x_train, rep, seed=0)

    h = mean_branching_entropy(model, log_base=2.0)
    assert h >= 0.0
    assert h < 10.0  # pretty loose upper bound