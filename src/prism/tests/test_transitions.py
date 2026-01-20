import math

from prism.processes import EvenProcess, GoldenMean
from prism.reconstruction import OneStepGreedyMerge
from prism.representations import LastK


def _fit_model(process, k: int, length: int, seed: int):
    rep = LastK(k=k)
    recon = OneStepGreedyMerge(eps=0.02)

    sample = process.sample(length=length, seed=seed)
    x = sample.x
    split = int(0.8 * len(x))
    x_train = x[:split]

    model = recon.fit(x_train, rep, seed=seed)
    return model


def test_transition_probabilities_sum_to_one():
    """
    For every (state, symbol) key, the outgoing next-state probabilities must sum to ~1.
    """
    process = EvenProcess(p_emit_one=0.7)
    model = _fit_model(process, k=3, length=50_000, seed=0)

    assert model.transitions, "No transitions were learned; something is wrong."

    for (s, sym), dist in model.transitions.items():
        assert sym in (0, 1)
        assert isinstance(dist, dict)
        assert len(dist) > 0, f"Empty transition distribution for (s={s}, sym={sym})."

        total = sum(dist.values())
        assert math.isfinite(total)
        assert abs(total - 1.0) < 1e-9, f"Transitions for (s={s}, sym={sym}) sum to {total}, not 1."

        for sp, p in dist.items():
            assert math.isfinite(p)
            assert 0.0 <= p <= 1.0, f"Invalid probability {p} for (s={s}, sym={sym}) -> {sp}."


def test_transitions_keys_consistent_with_states():
    """
    All states referenced in transitions should be integers, and
    next-states should be integers too.
    """
    process = GoldenMean(p_emit_one=0.7)
    model = _fit_model(process, k=3, length=50_000, seed=1)

    for (s, sym), dist in model.transitions.items():
        assert isinstance(s, int)
        assert isinstance(sym, int)
        for sp, p in dist.items():
            assert isinstance(sp, int)
            assert isinstance(p, float)