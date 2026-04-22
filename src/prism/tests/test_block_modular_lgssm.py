import numpy as np

from prism.processes.block_modular_lgssm import BlockModularLGSSM


def test_block_modular_generator_shapes_and_labels():
    process = BlockModularLGSSM(coupling=0.05, obs_dim=8, obs_design="random")
    sample = process.sample(length=32, seed=0)

    obs = np.asarray(sample.x, dtype=float)
    latent = np.asarray(sample.latent, dtype=float)
    regimes = process.regime_labels(latent)

    assert obs.shape == (32, 8)
    assert latent.shape == (32, 4)
    assert process.block_attribution(latent).shape == (32,)
    assert set(regimes) == {"slow_block", "phase_block", "joint"}
    assert all(values.shape == (32,) for values in regimes.values())
