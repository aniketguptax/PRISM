import math

import numpy as np

from prism.continuous.psi import InnovationsForm, compute_ss_psi, optimise_ss_psi
from prism.processes.linear_gaussian_ssm import LinearGaussianSSM
from prism.reconstruction.kalman_iss import KalmanISSReconstructor
from prism.representations.continuous import ISSDim


def _toy_innovations() -> InnovationsForm:
    A = np.array([[0.7, 0.1], [0.0, 0.65]], dtype=float)
    C = np.array([[1.0, 0.3], [0.2, 1.0]], dtype=float)
    V = np.array([[0.8, 0.1], [0.1, 0.7]], dtype=float)
    K = np.array([[0.4, 0.05], [0.03, 0.35]], dtype=float)
    return InnovationsForm(A=A, C=C, K=K, V=V)


def test_compute_ss_psi_finite() -> None:
    innovations = _toy_innovations()
    L = np.array([[1.0, -0.2]], dtype=float)
    psi = compute_ss_psi(innovations, L, tol=1e-8, max_iter=2000, ridge=1e-8)
    assert math.isfinite(psi)


def test_optimise_ss_psi_random_finite() -> None:
    innovations = _toy_innovations()
    result = optimise_ss_psi(
        innovations,
        macro_dim=1,
        seed=0,
        optimiser="random",
        restarts=3,
        iterations=20,
        step_scale=0.2,
        tol=1e-8,
        max_iter=1500,
        ridge=1e-8,
    )
    assert result.L.shape == (1, 2)
    assert math.isfinite(result.psi)


def test_kalman_reconstructor_compute_psi_populates_fields() -> None:
    process = LinearGaussianSSM(a=0.9, process_std=0.3, obs_std=0.2)
    sample = process.sample(length=800, seed=0)
    recon = KalmanISSReconstructor(
        em_iters=15,
        em_tol=1e-4,
        em_ridge=1e-6,
        compute_psi=True,
        psi_optimiser="random",
        psi_restarts=2,
        psi_iterations=20,
        psi_step_scale=0.2,
        psi_max_iter=1200,
    )
    model = recon.fit(sample.x[:600], ISSDim(d=2), seed=0)
    assert model.psi_opt is not None
    assert math.isfinite(model.psi_opt)
    assert model.psi_macro_dim == 1
    assert model.psi_optimiser == "random"
