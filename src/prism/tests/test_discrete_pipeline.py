import math

from prism.metrics.branching import mean_branching_entropy_weighted
from prism.metrics.predictive import log_loss_with_context
from prism.metrics.unifilarity import unifilarity_score
from prism.processes.iid import IIDBernoulli
from prism.processes.markov_order_1 import MarkovOrder1
from prism.reconstruction.one_step_merge import OneStepGreedyMerge
from prism.representations.discrete import LastK


def _fit_and_score(process, seed: int = 0) -> tuple[float, float, float]:
    rep = LastK(k=2)
    recon = OneStepGreedyMerge(eps=0.02)
    sample = process.sample(length=40_000, seed=seed)
    split = int(0.8 * len(sample.x))
    model = recon.fit(sample.x[:split], rep, seed=seed)
    ll = log_loss_with_context(sample.x[:split], sample.x[split:], rep, model)
    unif = unifilarity_score(model)
    branch = mean_branching_entropy_weighted(model, log_base=2.0)
    return ll, unif, branch


def test_iid_pipeline_metrics_finite() -> None:
    ll, unif, branch = _fit_and_score(IIDBernoulli(p_one=0.5), seed=1)
    assert math.isfinite(ll)
    assert 0.3 < ll < 1.5
    assert 0.0 <= unif <= 1.0
    assert branch >= 0.0


def test_markov_pipeline_metrics_finite() -> None:
    ll, unif, branch = _fit_and_score(MarkovOrder1(p01=0.8, p11=0.2), seed=2)
    assert math.isfinite(ll)
    assert 0.0 <= unif <= 1.0
    assert branch >= 0.0
