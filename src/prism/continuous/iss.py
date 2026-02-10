from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

Array = np.ndarray


def _sym(A: Array) -> Array:
    return 0.5 * (A + A.T)


def _ensure_spd(S: Array, *, ridge: float = 1e-9, max_tries: int = 8) -> Array:
    """Return a symmetric positive-definite matrix by adding adaptive jitter."""
    S_sym = _sym(S)
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
    """
    Returns inv(S) and logdet(S) robustly via Cholesky.
    """
    S_spd = _ensure_spd(S)
    L = np.linalg.cholesky(S_spd)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(S_spd.shape[0])))
    return inv, float(logdet)

def _gaussian_nll(x: Array, mean: Array, cov: Array) -> float:
    """
    NLL for a single observation x under N(mean, cov).
    """
    x = x.reshape(-1, 1)
    mean = mean.reshape(-1, 1)
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
    """
    Linear-Gaussian state-space model:
      z_{t+1} = A z_t + w_t,  w_t ~ N(0, Q)
      y_t     = C z_t + v_t,  v_t ~ N(0, R)
      z_0     ~ N(mu0, V0)

    This induces an innovations form; Kalman filter yields
    one-step predictive y_{t+1}|y_{1:t} with mean mu_t and covariance S_t,
    and innovations e_{t+1} = y_{t+1} - mu_t.
    """
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

def _init_params(y: Array, d: int, ridge: float, seed: int, init_state_cov: float, init_obs_cov: float) -> KalmanISSModel:
    """
    Safe initialisation:
    - C from first d PCs (if possible), else random small.
    - A near identity.
    - Q, R diagonal.
    """
    rng = np.random.default_rng(seed)
    T, p = y.shape

    # demean
    y0 = y - np.mean(y, axis=0, keepdims=True)

    # C init
    if T >= d and p >= 1:
        # PCA via SVD on y0
        _, _, Vt = np.linalg.svd(y0, full_matrices=False)
        # Vt: (p,p), take top-d directions
        V = Vt.T[:, :d] if Vt.shape[0] >= d else rng.normal(scale=0.1, size=(p, d))
        C = V
    else:
        C = rng.normal(scale=0.1, size=(p, d))

    A = np.eye(d) * 0.95  # stable-ish
    Q = np.eye(d) * float(init_state_cov)
    R = np.eye(p) * float(init_obs_cov)

    mu0 = np.zeros((d, 1))
    V0 = np.eye(d) * float(init_state_cov)

    # ridge stabilisation
    Q = _ensure_spd(Q, ridge=ridge)
    R = _ensure_spd(R, ridge=ridge)
    V0 = _ensure_spd(V0, ridge=ridge)

    return KalmanISSModel(A=A, C=C, Q=Q, R=R, mu0=mu0, V0=V0)

def kalman_filter(y: Array, m: KalmanISSModel) -> Tuple[Array, Array, Array, Array, float]:
    """
    Standard Kalman filter.

    Returns:
      mu_filt[t]  = E[z_t | y_0:t]
      P_filt[t]   = Cov[z_t | y_0:t]
      mu_pred[t]  = E[z_t | y_0:t-1]   (with mu_pred[0]=mu0)
      P_pred[t]   = Cov[z_t | y_0:t-1]
      ll          = log-likelihood of y under model
    """
    A, C, Q, R = m.A, m.C, _ensure_spd(m.Q), _ensure_spd(m.R)
    mu0, V0 = m.mu0, _ensure_spd(m.V0)

    T, p = y.shape
    d = A.shape[0]

    mu_f = np.zeros((T, d, 1))
    P_f = np.zeros((T, d, d))
    mu_pr = np.zeros((T, d, 1))
    P_pr = np.zeros((T, d, d))

    ll = 0.0

    # t=0 prediction
    mu_pr[0] = mu0
    P_pr[0] = V0

    I = np.eye(d)

    for t in range(T):
        # Predict observation
        y_mean = C @ mu_pr[t]             # (p,1)
        S = _ensure_spd(C @ P_pr[t] @ C.T + R)

        # Update
        invS, logdetS = _chol_inv(S)
        K = P_pr[t] @ C.T @ invS          # (d,p)
        innov = y[t].reshape(p, 1) - y_mean

        mu_f[t] = mu_pr[t] + K @ innov
        P_f[t] = _ensure_spd((I - K @ C) @ P_pr[t] @ (I - K @ C).T + K @ R @ K.T)

        # log-likelihood increment
        quad = float((innov.T @ invS @ innov).item())
        ll += -0.5 * (p * np.log(2.0 * np.pi) + logdetS + quad)

        # next predict
        if t + 1 < T:
            mu_pr[t + 1] = A @ mu_f[t]
            P_pr[t + 1] = _ensure_spd(A @ P_f[t] @ A.T + Q)

    return mu_f, P_f, mu_pr, P_pr, float(ll)

def rts_smoother(m: KalmanISSModel, mu_f: Array, P_f: Array, mu_pr: Array, P_pr: Array) -> Tuple[Array, Array, Array]:
    """
    Rauch–Tung–Striebel smoother.

    Returns:
      mu_s[t] = E[z_t | y_0:T-1]
      P_s[t]  = Cov[z_t | y_0:T-1]
      P_cs[t] = Cov[z_t, z_{t-1} | y] for t>=1, shape (T,d,d), P_cs[0]=0
    """
    A = m.A

    T, d, _ = mu_f.shape
    mu_s = mu_f.copy()
    P_s = P_f.copy()
    P_cs = np.zeros((T, d, d))

    for t in range(T - 2, -1, -1):
        Ppr = _ensure_spd(P_pr[t + 1])
        J = P_f[t] @ A.T @ np.linalg.solve(Ppr, np.eye(d))  # smoother gain

        mu_s[t] = mu_f[t] + J @ (mu_s[t + 1] - mu_pr[t + 1])
        P_s[t] = _ensure_spd(P_f[t] + J @ (P_s[t + 1] - Ppr) @ J.T)

        # Cross-covariance (t+1,t)
        P_cs[t + 1] = _sym(P_s[t + 1] @ J.T)

    return mu_s, P_s, P_cs

def fit_kalman_iss_em(y: Array, cfg: KalmanISSConfig) -> KalmanISSModel:
    """
    EM for linear-Gaussian SSM (Kalman ISS).
    Fully general (multivariate y), robust with ridge.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    T, p = y.shape
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

        # Expectations
        # E[z_t] = mu_s[t]
        # E[z_t z_t^T] = P_s[t] + mu_s[t] mu_s[t]^T
        Ez = mu_s
        Ezz = P_s + np.einsum("tij,tkj->tik", mu_s, mu_s)  # (T,d,d)

        # E[z_t z_{t-1}^T] for t>=1: P_cs[t] + mu_s[t] mu_s[t-1]^T
        Ezzm1 = np.zeros_like(P_cs)
        for t in range(1, T):
            Ezzm1[t] = P_cs[t] + (mu_s[t] @ mu_s[t - 1].T)

        # M-step: update A, Q
        sum_Ezz_prev = np.sum(Ezz[:-1], axis=0)  # t=0..T-2
        sum_Ezzm1 = np.sum(Ezzm1[1:], axis=0)    # t=1..T-1 cross

        # Regularise inversion
        sum_Ezz_prev_reg = _ensure_spd(sum_Ezz_prev, ridge=cfg.ridge)
        A_new = np.linalg.solve(sum_Ezz_prev_reg.T, sum_Ezzm1.T).T

        # Q = (1/(T-1)) * sum E[(z_{t} - A z_{t-1})(...)]
        Q_num = np.zeros((d, d))
        for t in range(1, T):
            Q_num += Ezz[t] - A_new @ Ezzm1[t].T - Ezzm1[t] @ A_new.T + A_new @ Ezz[t - 1] @ A_new.T
        Q_new = _ensure_spd(Q_num / max(T - 1, 1), ridge=cfg.ridge)

        # Update C, R
        # C = (sum y_t E[z_t]^T) (sum E[z_t z_t^T])^{-1}
        sum_yz = np.zeros((p, d))
        for t in range(T):
            sum_yz += y[t].reshape(p, 1) @ Ez[t].T

        sum_Ezz_all = np.sum(Ezz, axis=0)
        sum_Ezz_all_reg = _ensure_spd(sum_Ezz_all, ridge=cfg.ridge)
        C_new = np.linalg.solve(sum_Ezz_all_reg.T, sum_yz.T).T

        # R = (1/T) sum E[(y_t - C z_t)(...)]
        R_num = np.zeros((p, p))
        for t in range(T):
            yt = y[t].reshape(p, 1)
            R_num += yt @ yt.T - C_new @ Ez[t] @ yt.T - yt @ Ez[t].T @ C_new.T + C_new @ Ezz[t] @ C_new.T
        R_new = _ensure_spd(R_num / max(T, 1), ridge=cfg.ridge)

        # Initial state
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

        # Convergence
        if it > 0 and abs(ll - prev_ll) < cfg.tol * (1.0 + abs(prev_ll)):
            break
        prev_ll = ll

    return model

def one_step_predictive_y(y: Array, model: KalmanISSModel) -> Tuple[Array, Array, Array]:
    """
    Using the Kalman filter on y, return:
      mu_y[t] = E[y_{t} | y_{0:t-1}]    (one-step predictive mean)
      S_y[t]  = Cov[y_{t} | y_{0:t-1}]  (one-step predictive cov)
      innov[t]= y_t - mu_y[t]
    with mu_y[0]=C mu0, S_y[0]=C V0 C^T + R.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    T, p = y.shape

    mu_f, P_f, mu_pr, P_pr, _ = kalman_filter(y, model)

    mu_y = np.zeros((T, p, 1))
    S_y = np.zeros((T, p, p))
    innov = np.zeros((T, p, 1))

    C, R = model.C, _ensure_spd(model.R)

    for t in range(T):
        mu_y[t] = C @ mu_pr[t]
        S_y[t] = _ensure_spd(C @ P_pr[t] @ C.T + R)
        innov[t] = y[t].reshape(p, 1) - mu_y[t]

    return mu_y, S_y, innov
