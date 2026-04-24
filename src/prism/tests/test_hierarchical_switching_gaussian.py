import numpy as np

from prism.processes.hierarchical_switching_gaussian import HierarchicalSwitchingGaussian


def test_hierarchical_switching_gaussian_shapes_and_regimes():
    process = HierarchicalSwitchingGaussian(obs_dim=8, emission_std=0.2)
    sample = process.sample(length=64, seed=0)

    obs = np.asarray(sample.x, dtype=float)
    latent = np.asarray(sample.latent, dtype=int)
    regimes = process.regime_labels(latent)

    assert obs.shape == (64, 8)
    assert latent.shape == (64, 3)
    assert set(regimes) == {"coarse", "fine", "joint"}
    assert all(values.shape == (64,) for values in regimes.values())
    assert np.array_equal(regimes["joint"], regimes["coarse"] * process.n_fine + regimes["fine"])
