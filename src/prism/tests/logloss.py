import math

from prism.metrics.predictive import log_loss_with_context
from prism.processes import IIDBernoulli
from prism.reconstruction import OneStepGreedyMerge
from prism.representations import LastK


def test_logloss_finite_and_reasonable():
    process = IIDBernoulli(p_one=0.5)
    rep = LastK(k=2)
    recon = OneStepGreedyMerge(eps=0.02)

    sample = process.sample(length=60_000, seed=0)
    x = sample.x
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]

    model = recon.fit(x_train, rep, seed=0)
    ll = log_loss_with_context(x_train, x_test, rep, model)

    assert math.isfinite(ll)
    # For fair coin, cross-entropy should be close to ln(2) ~0.693 (this is just sanity)
    assert 0.4 < ll < 1.2