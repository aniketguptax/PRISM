from prism.metrics.unifilarity import unifilarity_score
from prism.processes import EvenProcess
from prism.reconstruction import OneStepGreedyMerge
from prism.representations import LastK


def test_unifilarity_in_range():
    process = EvenProcess(p_emit_one=0.7)
    rep = LastK(k=3)
    recon = OneStepGreedyMerge(eps=0.02)

    x = process.sample(length=50_000, seed=0).x
    x_train = x[: int(0.8 * len(x))]
    model = recon.fit(x_train, rep, seed=0)

    u = unifilarity_score(model)
    assert 0.0 <= u <= 1.0