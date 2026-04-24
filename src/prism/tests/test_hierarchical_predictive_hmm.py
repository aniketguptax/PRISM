import numpy as np

from prism.processes.hierarchical_predictive_hmm import HierarchicalPredictiveHMM


def test_hierarchical_predictive_hmm_shapes_and_regimes():
    process = HierarchicalPredictiveHMM(emission_noise=0.08)
    sample = process.sample(length=64, seed=0)

    obs = np.asarray(sample.x, dtype=int)
    latent = np.asarray(sample.latent, dtype=int)
    regimes = process.regime_labels(latent)

    assert obs.shape == (64,)
    assert latent.shape == (64, 3)
    assert obs.min() >= 0
    assert obs.max() < process.alphabet_size
    assert set(regimes) == {"coarse", "fine", "joint"}
    assert all(values.shape == (64,) for values in regimes.values())
    assert np.array_equal(regimes["joint"], regimes["coarse"] * process.n_fine + regimes["fine"])
