import numpy as np

from prism.processes.predictive_low_variance_lgssm import PredictiveLowVarianceLGSSM


def test_predictive_low_variance_lgssm_shapes_and_regimes():
    process = PredictiveLowVarianceLGSSM(obs_dim=8, latent_dim=3)
    sample = process.sample(length=64, seed=0)

    obs = np.asarray(sample.x, dtype=float)
    latent = np.asarray(sample.latent, dtype=float)
    regimes = process.regime_labels(latent)

    assert obs.shape == (64, 8)
    assert latent.shape == (64, 3)
    assert set(regimes) == {"slow", "fast", "joint"}
    assert all(values.shape == (64,) for values in regimes.values())
