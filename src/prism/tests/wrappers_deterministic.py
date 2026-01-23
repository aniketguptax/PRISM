from prism.processes import EvenProcess
from prism.processes.wrappers import NoisyObservation, Subsample


def test_noisy_observation_deterministic_given_seed():
    base = EvenProcess(p_emit_one=0.7)
    proc = NoisyObservation(base=base, flip_p=0.2)

    s1 = proc.sample(length=10_000, seed=123)
    s2 = proc.sample(length=10_000, seed=123)

    assert s1.x == s2.x
    assert s1.latent == s2.latent


def test_subsample_deterministic_given_seed():
    base = EvenProcess(p_emit_one=0.7)
    proc = Subsample(base=base, step=3)

    s1 = proc.sample(length=10_000, seed=999)
    s2 = proc.sample(length=10_000, seed=999)

    assert s1.x == s2.x
    assert s1.latent == s2.latent


def test_composed_wrappers_deterministic_given_seed():
    base = EvenProcess(p_emit_one=0.7)
    proc = Subsample(base=NoisyObservation(base=base, flip_p=0.1), step=2)

    s1 = proc.sample(length=10_000, seed=42)
    s2 = proc.sample(length=10_000, seed=42)

    assert s1.x == s2.x
    assert s1.latent == s2.latent