from __future__ import annotations

import numpy as np

from prism.processes.multiscale_lgssm import MultiscaleLGSSM


def test_multiscale_lgssm_sample_shapes_and_labels() -> None:
    process = MultiscaleLGSSM(obs_dim=8)
    sample = process.sample(128, seed=7)

    obs = np.asarray(sample.x, dtype=float)
    latent = np.asarray(sample.latent, dtype=float)
    labels = process.regime_labels(latent, slow_bins=3, phase_bins=4)

    assert obs.shape == (128, 8)
    assert latent.shape == (128, 5)
    assert set(labels) == {"slow", "phase", "joint"}
    assert labels["slow"].min() >= 0
    assert labels["slow"].max() < 3
    assert labels["phase"].min() >= 0
    assert labels["phase"].max() < 4
    assert labels["joint"].min() >= 0
    assert labels["joint"].max() < 12


def test_multiscale_lgssm_is_reproducible() -> None:
    process = MultiscaleLGSSM(obs_dim=8)
    first = process.sample(64, seed=3)
    second = process.sample(64, seed=3)

    np.testing.assert_allclose(np.asarray(first.x, dtype=float), np.asarray(second.x, dtype=float))
    np.testing.assert_allclose(np.asarray(first.latent, dtype=float), np.asarray(second.latent, dtype=float))
