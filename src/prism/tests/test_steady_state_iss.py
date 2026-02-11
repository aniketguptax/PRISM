import numpy as np

from prism.continuous.iss import KalmanISSModel, iss_filter, solve_steady_state_kalman
from prism.processes.linear_gaussian_ssm import LinearGaussianSSM
from prism.reconstruction.kalman_iss import KalmanISSReconstructor
from prism.representations.continuous import ISSDim


def _simulate_scalar_ssm(model: KalmanISSModel, length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = np.zeros((length, 1), dtype=float)
    y = np.zeros((length, 1), dtype=float)
    q_std = float(np.sqrt(model.Q[0, 0]))
    r_std = float(np.sqrt(model.R[0, 0]))

    z[0, 0] = float(rng.normal(0.0, 1.0))
    y[0, 0] = float((model.C @ z[0].reshape(1, 1)).item() + rng.normal(0.0, r_std))
    for t in range(1, length):
        z[t, 0] = float((model.A @ z[t - 1].reshape(1, 1)).item() + rng.normal(0.0, q_std))
        y[t, 0] = float((model.C @ z[t].reshape(1, 1)).item() + rng.normal(0.0, r_std))
    return y


def test_steady_state_gain_is_constant_in_iss_filter() -> None:
    model = KalmanISSModel(
        A=np.array([[0.85]], dtype=float),
        C=np.array([[1.0]], dtype=float),
        Q=np.array([[0.15]], dtype=float),
        R=np.array([[0.2]], dtype=float),
        mu0=np.array([[0.0]], dtype=float),
        V0=np.array([[1.0]], dtype=float),
    )
    y = _simulate_scalar_ssm(model, length=120, seed=3)

    steady = solve_steady_state_kalman(model, tol=1e-10, max_iter=20_000, ridge=1e-10, strict=True)
    assert steady.converged

    mu_f, _, _, p_pr, _ = iss_filter(
        y,
        model,
        steady_state=True,
        steady_state_tol=1e-10,
        steady_state_max_iter=20_000,
        steady_state_ridge=1e-10,
        steady_state_solution=steady,
    )

    # Fixed-gain recursion implies the inferred gain from update residuals is constant.
    gains: list[float] = []
    for t in range(1, y.shape[0]):
        pred = model.A @ mu_f[t - 1]
        innov = y[t].reshape(1, 1) - model.C @ pred
        if abs(float(innov.item())) < 1e-9:
            continue
        gains.append(float(((mu_f[t] - pred) / innov).item()))

    assert gains
    assert np.allclose(np.asarray(gains), steady.correction_gain.item(), atol=1e-8, rtol=1e-6)
    assert np.allclose(p_pr, steady.pred_error_cov[None, :, :], atol=1e-12, rtol=0.0)


def test_reconstructor_defaults_to_strict_steady_state_mode() -> None:
    process = LinearGaussianSSM(latent_dim=2, obs_dim=3, a=0.9, process_std=0.3, obs_std=0.2)
    sample = process.sample(length=1000, seed=5)
    split = 800
    x_train = sample.x[:split]
    x_test = sample.x[split:]

    recon = KalmanISSReconstructor(
        em_iters=20,
        em_tol=1e-4,
        em_ridge=1e-6,
        macro_eps=0.25,
        macro_bins=3,
    )
    model = recon.fit(x_train, ISSDim(d=2, dv=2), seed=0)
    assert model.iss_mode == "steady_state"

    _, covs = model.predictive_distributions(x_test, context=x_train)
    assert covs.shape[0] == len(x_test)
    assert np.allclose(covs, covs[0][None, :, :], atol=1e-8, rtol=0.0)
