from prism.processes import EvenProcess
from prism.reconstruction import OneStepGreedyMerge
from prism.representations import LastK


def _learn(seed: int):
    process = EvenProcess(p_emit_one=0.7)
    rep = LastK(k=3)
    recon = OneStepGreedyMerge(eps=0.02)

    sample = process.sample(length=80_000, seed=seed)
    x = sample.x
    split = int(0.8 * len(x))
    x_train = x[:split]

    model = recon.fit(x_train, rep, seed=seed)

    # Compare only pure-data structures (dict ordering is stable in Py3.7+)
    return {
        "rep_to_state": model.rep_to_state,
        "p_next_one": model.p_next_one,
        "pi": model.pi,
        "transitions": model.transitions,
    }


def test_same_seed_same_model():
    m1 = _learn(seed=0)
    m2 = _learn(seed=0)
    assert m1 == m2


def test_different_seed_usually_differs():
    """
    Not a mathematical guarantee, but on a stochastic process this should almost always differ.
    If this fails, seeding might be ignored, or there is an accidental global RNG.
    """
    m1 = _learn(seed=0)
    m2 = _learn(seed=1)
    assert m1 != m2