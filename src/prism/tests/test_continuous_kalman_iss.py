import math

from prism.metrics.gaussian_predictive import gaussian_log_loss
from prism.processes.linear_gaussian_ssm import LinearGaussianSSM
from prism.reconstruction.kalman_iss import KalmanISSReconstructor
from prism.representations.continuous import ISSDim


def test_kalman_iss_fits_and_scores_finite() -> None:
    process = LinearGaussianSSM(a=0.9, process_std=0.3, obs_std=0.2)
    sample = process.sample(length=1200, seed=0)
    split = 900
    x_train = sample.x[:split]
    x_test = sample.x[split:]

    recon = KalmanISSReconstructor(em_iters=25, em_tol=1e-4, em_ridge=1e-6)
    model = recon.fit(x_train, ISSDim(d=2), seed=0)
    ll = gaussian_log_loss(x_test, model, context=x_train)

    assert model.latent_dim == 2
    assert math.isfinite(ll)
    assert ll > 0.0

    filtered_means, filtered_covs = model.filtered_state_sequence(x_test, context=x_train)
    assert filtered_means.shape[0] == len(x_test)
    assert filtered_covs.shape[0] == len(x_test)
