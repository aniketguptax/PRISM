from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import warnings

import numpy as np

Array = np.ndarray


def _sym(A: Array) -> Array:
    return 0.5 * (A + A.T)


def _ensure_spd(S: Array, *, ridge: float = 1e-9, max_tries: int = 8) -> Array:
    """Return a symmetric positive-definite matrix by adding adaptive jitter."""
    S_sym = _sym(np.asarray(S, dtype=float))
    eye = np.eye(S_sym.shape[0], dtype=float)
    jitter = float(max(ridge, 0.0))
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(S_sym + jitter * eye)
            return S_sym + jitter * eye
        except np.linalg.LinAlgError:
            jitter = 10.0 * jitter if jitter > 0.0 else 1e-12
    raise np.linalg.LinAlgError("Failed to make covariance matrix positive definite.")


def _chol_inv(S: Array) -> Tuple[Array, float]:
    """Return ``(S^{-1}, logdet(S))`` robustly via Cholesky."""
    S_spd = _ensure_spd(S)
    L = np.linalg.cholesky(S_spd)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(S_spd.shape[0])))
    return inv, float(logdet)


def _gaussian_nll(x: Array, mean: Array, cov: Array) -> float:
    """NLL for one observation under ``N(mean, cov)``."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    mean = np.asarray(mean, dtype=float).reshape(-1, 1)
    cov = _ensure_spd(cov)
    inv, logdet = _chol_inv(cov)
    d = x.shape[0]
    diff = x - mean
    quad = float((diff.T @ inv @ diff).item())
    return 0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


@dataclass
class KalmanISSConfig:
    latent_dim: int
    em_iters: int = 50
    tol: float = 1e-4
    ridge: float = 1e-6
    init_state_cov: float = 1.0
    init_obs_cov: float = 1.0
    seed: int = 0


@dataclass
class KalmanISSModel:
    """Linear-Gaussian state-space model parameters."""

    A: Array
    C: Array
    Q: Array
    R: Array
    mu0: Array
    V0: Array

    def copy(self) -> "KalmanISSModel":
        return KalmanISSModel(
            A=self.A.copy(),
            C=self.C.copy(),
            Q=self.Q.copy(),
            R=self.R.copy(),
            mu0=self.mu0.copy(),
            V0=self.V0.copy(),
        )


@dataclass(frozen=True)
class SteadyStateKalmanSolution:
    """Steady-state filtering quantities for the ISS recursion.

    ``correction_gain`` is the fixed gain ``G`` in
    ``S_t = z_{t|t-1} + G (y_t - C z_{t|t-1})``.
    ``innovations_gain`` is ``A G`` for the innovations form
    ``x_{t+1} = A x_t + K e_t`` with prediction-state convention.
    """

    pred_error_cov: Array
    filt_error_cov: Array
    innovation_cov: Array
    correction_gain: Array
    innovations_gain: Array
    iterations: int
    converged: bool
    rel_change: float


def _as_matrix_obs(y: Array) -> Array:
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        return y_arr.reshape(-1, 1)
    if y_arr.ndim != 2:
        raise ValueError(f"Expected observations with shape (T, p), got {y_arr.shape}.")
    return y_arr


def _validate_model_shapes(model: KalmanISSModel, obs_dim: int) -> tuple[int, int]:
    A = np.asarray(model.A, dtype=float)
    C = np.asarray(model.C, dtype=float)
    Q = np.asarray(model.Q, dtype=float)
    R = np.asarray(model.R, dtype=float)
    mu0 = np.asarray(model.mu0, dtype=float)
    V0 = np.asarray(model.V0, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}.")
    d = A.shape[0]
    if C.shape != (obs_dim, d):
        raise ValueError(f"C must have shape ({obs_dim}, {d}), got {C.shape}.")
    if Q.shape != (d, d):
        raise ValueError(f"Q must have shape ({d}, {d}), got {Q.shape}.")
    if R.shape != (obs_dim, obs_dim):
        raise ValueError(f"R must have shape ({obs_dim}, {obs_dim}), got {R.shape}.")
    if mu0.shape != (d, 1):
        raise ValueError(f"mu0 must have shape ({d}, 1), got {mu0.shape}.")
    if V0.shape != (d, d):
        raise ValueError(f"V0 must have shape ({d}, {d}), got {V0.shape}.")
    return d, obs_dim


def _init_params(
    y: Array,
    d: int,
    ridge: float,
    seed: int,
    init_state_cov: float,
    init_obs_cov: float,
) -> KalmanISSModel:
    """Safe initialisation for EM."""
    rng = np.random.default_rng(seed)
    T, p = y.shape

    y0 = y - np.mean(y, axis=0, keepdims=True)

    if T >= d and p >= 1:
        _, _, Vt = np.linalg.svd(y0, full_matrices=False)
        V = Vt.T[:, :d] if Vt.shape[0] >= d else rng.normal(scale=0.1, size=(p, d))
        C = V
    else:
        C = rng.normal(scale=0.1, size=(p, d))

    A = np.eye(d) * 0.95
    Q = np.eye(d) * float(init_state_cov)
    R = np.eye(p) * float(init_obs_cov)

    mu0 = np.zeros((d, 1))
    V0 = np.eye(d) * float(init_state_cov)

    Q = _ensure_spd(Q, ridge=ridge)
    R = _ensure_spd(R, ridge=ridge)
    V0 = _ensure_spd(V0, ridge=ridge)

    return KalmanISSModel(A=A, C=C, Q=Q, R=R, mu0=mu0, V0=V0)


def kalman_filter(y: Array, m: KalmanISSModel) -> Tuple[Array, Array, Array, Array, float]:
    """Standard time-varying-gain Kalman filter."""
    y = _as_matrix_obs(y)
    T, p = y.shape

    d, _ = _validate_model_shapes(m, obs_dim=p)
    A, C = np.asarray(m.A, dtype=float), np.asarray(m.C, dtype=float)
    Q, R = _ensure_spd(m.Q), _ensure_spd(m.R)
    mu0, V0 = np.asarray(m.mu0, dtype=float), _ensure_spd(m.V0)

    mu_f = np.zeros((T, d, 1))
    P_f = np.zeros((T, d, d))
    mu_pr = np.zeros((T, d, 1))
    P_pr = np.zeros((T, d, d))

    ll = 0.0
    if T == 0:
        return mu_f, P_f, mu_pr, P_pr, ll

    mu_pr[0] = mu0
    P_pr[0] = V0

    I = np.eye(d)

    for t in range(T):
        y_mean = C @ mu_pr[t]
        S = _ensure_spd(C @ P_pr[t] @ C.T + R)

        invS, logdetS = _chol_inv(S)
        K_t = P_pr[t] @ C.T @ invS
        innov = y[t].reshape(p, 1) - y_mean

        mu_f[t] = mu_pr[t] + K_t @ innov
        P_f[t] = _ensure_spd((I - K_t @ C) @ P_pr[t] @ (I - K_t @ C).T + K_t @ R @ K_t.T)

        quad = float((innov.T @ invS @ innov).item())
        ll += -0.5 * (p * np.log(2.0 * np.pi) + logdetS + quad)

        if t + 1 < T:
            mu_pr[t + 1] = A @ mu_f[t]
            P_pr[t + 1] = _ensure_spd(A @ P_f[t] @ A.T + Q)

    return mu_f, P_f, mu_pr, P_pr, float(ll)


def solve_steady_state_kalman(
    model: KalmanISSModel,
    *,
    tol: float = 1e-9,
    max_iter: int = 10_000,
    ridge: float = 1e-9,
    strict: bool = True,
) -> SteadyStateKalmanSolution:
    """To solve the steady-state filtering DARE and return fixed ISS gains.

    The solved covariance is the prediction-error covariance ``P = Cov[z_t - z_{t|t-1}]``.
    """
    if tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")

    A = np.asarray(model.A, dtype=float)
    C = np.asarray(model.C, dtype=float)
    Q = _ensure_spd(np.asarray(model.Q, dtype=float), ridge=ridge)
    R = _ensure_spd(np.asarray(model.R, dtype=float), ridge=ridge)
    V0 = _ensure_spd(np.asarray(model.V0, dtype=float), ridge=ridge)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}.")
    d = A.shape[0]
    p = C.shape[0]
    if C.shape[1] != d:
        raise ValueError(f"C must have {d} columns, got shape {C.shape}.")
    if Q.shape != (d, d):
        raise ValueError(f"Q must have shape ({d}, {d}), got {Q.shape}.")
    if R.shape != (p, p):
        raise ValueError(f"R must have shape ({p}, {p}), got {R.shape}.")

    I = np.eye(d, dtype=float)
    P = _ensure_spd(A @ V0 @ A.T + Q, ridge=ridge)

    converged = False
    rel_change = np.inf
    iterations = 0

    for it in range(1, max_iter + 1):
        S = _ensure_spd(C @ P @ C.T + R, ridge=ridge)
        invS = np.linalg.solve(S, np.eye(p, dtype=float))
        G = P @ C.T @ invS
        P_f = _ensure_spd((I - G @ C) @ P @ (I - G @ C).T + G @ R @ G.T, ridge=ridge)
        P_next = _ensure_spd(A @ P_f @ A.T + Q, ridge=ridge)

        rel_change = np.linalg.norm(P_next - P, ord="fro") / max(np.linalg.norm(P, ord="fro"), 1e-12)
        P = P_next
        iterations = it
        if rel_change < tol:
            converged = True
            break

    S = _ensure_spd(C @ P @ C.T + R, ridge=ridge)
    invS = np.linalg.solve(S, np.eye(p, dtype=float))
    G = P @ C.T @ invS
    P_f = _ensure_spd((I - G @ C) @ P @ (I - G @ C).T + G @ R @ G.T, ridge=ridge)

    if not converged:
        message = (
            "Steady-state Kalman solver did not converge "
            f"within {max_iter} iterations (final relative change={rel_change:.3e})."
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning)

    return SteadyStateKalmanSolution(
        pred_error_cov=P,
        filt_error_cov=P_f,
        innovation_cov=S,
        correction_gain=G,
        innovations_gain=A @ G,
        iterations=iterations,
        converged=converged,
        rel_change=float(rel_change),
    )


def steady_state_kalman_filter(
    y: Array,
    model: KalmanISSModel,
    *,
    tol: float = 1e-9,
    max_iter: int = 10_000,
    ridge: float = 1e-9,
    strict: bool = True,
    solution: SteadyStateKalmanSolution | None = None,
) -> tuple[Array, Array, Array, Array, float, SteadyStateKalmanSolution]:
    """Run a fixed-gain steady-state ISS filter."""
    y = _as_matrix_obs(y)
    T, p = y.shape
    d, _ = _validate_model_shapes(model, obs_dim=p)

    if solution is None:
        solution = solve_steady_state_kalman(
            model,
            tol=tol,
            max_iter=max_iter,
            ridge=ridge,
            strict=strict,
        )

    A = np.asarray(model.A, dtype=float)
    C = np.asarray(model.C, dtype=float)
    mu0 = np.asarray(model.mu0, dtype=float)

    mu_f = np.zeros((T, d, 1), dtype=float)
    P_f = np.zeros((T, d, d), dtype=float)
    mu_pr = np.zeros((T, d, 1), dtype=float)
    P_pr = np.zeros((T, d, d), dtype=float)

    if T == 0:
        return mu_f, P_f, mu_pr, P_pr, 0.0, solution

    invS, logdetS = _chol_inv(solution.innovation_cov)
    ll = 0.0

    mu_pr[0] = mu0
    for t in range(T):
        P_pr[t] = solution.pred_error_cov
        y_mean = C @ mu_pr[t]
        innov = y[t].reshape(p, 1) - y_mean

        mu_f[t] = mu_pr[t] + solution.correction_gain @ innov
        P_f[t] = solution.filt_error_cov

        quad = float((innov.T @ invS @ innov).item())
        ll += -0.5 * (p * np.log(2.0 * np.pi) + logdetS + quad)

        if t + 1 < T:
            mu_pr[t + 1] = A @ mu_f[t]

    return mu_f, P_f, mu_pr, P_pr, float(ll), solution


def iss_filter(
    y: Array,
    model: KalmanISSModel,
    *,
    steady_state: bool = True,
    steady_state_tol: float = 1e-9,
    steady_state_max_iter: int = 10_000,
    steady_state_ridge: float = 1e-9,
    steady_state_strict: bool = True,
    allow_time_varying_fallback: bool = False,
    steady_state_solution: SteadyStateKalmanSolution | None = None,
) -> Tuple[Array, Array, Array, Array, float]:
    """Filter observations using steady-state ISS (default) or time-varying KF."""
    y = _as_matrix_obs(y)
    if steady_state:
        try:
            mu_f, P_f, mu_pr, P_pr, ll, _ = steady_state_kalman_filter(
                y,
                model,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
                ridge=steady_state_ridge,
                strict=steady_state_strict,
                solution=steady_state_solution,
            )
            return mu_f, P_f, mu_pr, P_pr, ll
        except Exception as exc:
            if not allow_time_varying_fallback:
                raise
            warnings.warn(
                "Steady-state ISS filtering failed; falling back to time-varying Kalman filter "
                f"({exc}).",
                RuntimeWarning,
            )
    return kalman_filter(y, model)


def rts_smoother(m: KalmanISSModel, mu_f: Array, P_f: Array, mu_pr: Array, P_pr: Array) -> Tuple[Array, Array, Array]:
    """Rauch-Tung-Striebel smoother"""
    A = np.asarray(m.A, dtype=float)

    T, d, _ = mu_f.shape
    mu_s = mu_f.copy()
    P_s = P_f.copy()
    P_cs = np.zeros((T, d, d))

    for t in range(T - 2, -1, -1):
        Ppr = _ensure_spd(P_pr[t + 1])
        J = P_f[t] @ A.T @ np.linalg.solve(Ppr, np.eye(d))

        mu_s[t] = mu_f[t] + J @ (mu_s[t + 1] - mu_pr[t + 1])
        P_s[t] = _ensure_spd(P_f[t] + J @ (P_s[t + 1] - Ppr) @ J.T)
        P_cs[t + 1] = _sym(P_s[t + 1] @ J.T)

    return mu_s, P_s, P_cs


def fit_kalman_iss_em(y: Array, cfg: KalmanISSConfig) -> KalmanISSModel:
    """EM for linear-Gaussian SSM"""
    y = _as_matrix_obs(y)
    T, _ = y.shape
    if T < 5:
        raise ValueError("Need at least 5 samples to fit Kalman ISS")

    d = int(cfg.latent_dim)
    if d < 1:
        raise ValueError("latent_dim must be >= 1")

    model = _init_params(
        y=y,
        d=d,
        ridge=cfg.ridge,
        seed=cfg.seed,
        init_state_cov=cfg.init_state_cov,
        init_obs_cov=cfg.init_obs_cov,
    )

    prev_ll = -np.inf
    for it in range(int(cfg.em_iters)):
        mu_f, P_f, mu_pr, P_pr, ll = kalman_filter(y, model)
        mu_s, P_s, P_cs = rts_smoother(model, mu_f, P_f, mu_pr, P_pr)

        Ez = mu_s
        Ezz = P_s + np.einsum("tij,tkj->tik", mu_s, mu_s)

        Ezzm1 = np.zeros_like(P_cs)
        for t in range(1, T):
            Ezzm1[t] = P_cs[t] + (mu_s[t] @ mu_s[t - 1].T)

        sum_Ezz_prev = np.sum(Ezz[:-1], axis=0)
        sum_Ezzm1 = np.sum(Ezzm1[1:], axis=0)

        sum_Ezz_prev_reg = _ensure_spd(sum_Ezz_prev, ridge=cfg.ridge)
        A_new = np.linalg.solve(sum_Ezz_prev_reg.T, sum_Ezzm1.T).T

        Q_num = np.zeros((d, d))
        for t in range(1, T):
            Q_num += Ezz[t] - A_new @ Ezzm1[t].T - Ezzm1[t] @ A_new.T + A_new @ Ezz[t - 1] @ A_new.T
        Q_new = _ensure_spd(Q_num / max(T - 1, 1), ridge=cfg.ridge)

        p = y.shape[1]
        sum_yz = np.zeros((p, d))
        for t in range(T):
            sum_yz += y[t].reshape(p, 1) @ Ez[t].T

        sum_Ezz_all = np.sum(Ezz, axis=0)
        sum_Ezz_all_reg = _ensure_spd(sum_Ezz_all, ridge=cfg.ridge)
        C_new = np.linalg.solve(sum_Ezz_all_reg.T, sum_yz.T).T

        R_num = np.zeros((p, p))
        for t in range(T):
            yt = y[t].reshape(p, 1)
            R_num += yt @ yt.T - C_new @ Ez[t] @ yt.T - yt @ Ez[t].T @ C_new.T + C_new @ Ezz[t] @ C_new.T
        R_new = _ensure_spd(R_num / max(T, 1), ridge=cfg.ridge)

        mu0_new = Ez[0].copy()
        V0_new = _ensure_spd(P_s[0], ridge=cfg.ridge)

        model = KalmanISSModel(
            A=A_new,
            C=C_new,
            Q=Q_new,
            R=R_new,
            mu0=mu0_new,
            V0=V0_new,
        )

        if it > 0 and abs(ll - prev_ll) < cfg.tol * (1.0 + abs(prev_ll)):
            break
        prev_ll = ll

    return model


def one_step_predictive_y(
    y: Array,
    model: KalmanISSModel,
    *,
    steady_state: bool = True,
    steady_state_tol: float = 1e-9,
    steady_state_max_iter: int = 10_000,
    steady_state_ridge: float = 1e-9,
    steady_state_strict: bool = True,
    allow_time_varying_fallback: bool = False,
    steady_state_solution: SteadyStateKalmanSolution | None = None,
) -> Tuple[Array, Array, Array]:
    """
    Return one-step predictive means/covariances and innovations.

    By default this uses strict steady-state ISS filtering with fixed gain.
    """
    y = _as_matrix_obs(y)
    T, p = y.shape

    _, _, mu_pr, P_pr, _ = iss_filter(
        y,
        model,
        steady_state=steady_state,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        steady_state_ridge=steady_state_ridge,
        steady_state_strict=steady_state_strict,
        allow_time_varying_fallback=allow_time_varying_fallback,
        steady_state_solution=steady_state_solution,
    )

    mu_y = np.zeros((T, p, 1), dtype=float)
    S_y = np.zeros((T, p, p), dtype=float)
    innov = np.zeros((T, p, 1), dtype=float)

    C, R = np.asarray(model.C, dtype=float), _ensure_spd(model.R)

    for t in range(T):
        mu_y[t] = C @ mu_pr[t]
        S_y[t] = _ensure_spd(C @ P_pr[t] @ C.T + R)
        innov[t] = y[t].reshape(p, 1) - mu_y[t]

    return mu_y, S_y, innov
