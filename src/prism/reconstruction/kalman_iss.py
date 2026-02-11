from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from prism.continuous.iss import (
    KalmanISSConfig,
    KalmanISSModel,
    SteadyStateKalmanSolution,
    _gaussian_nll,
    fit_kalman_iss_em,
    iss_filter,
    one_step_predictive_y,
    solve_steady_state_kalman,
)
from prism.continuous.psi import compute_ss_psi, innovations_from_ssm, optimise_ss_psi
from prism.representations.continuous import ISSDim
from prism.representations.protocols import Representation
from prism.types import Obs

from .protocols import Reconstructor

Array = np.ndarray


def _sym(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)


def _ensure_spd(matrix: Array, *, ridge: float = 1e-9, max_tries: int = 8) -> Array:
    out = _sym(np.asarray(matrix, dtype=float))
    eye = np.eye(out.shape[0], dtype=float)
    jitter = max(float(ridge), 0.0)
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(out + jitter * eye)
            return out + jitter * eye
        except np.linalg.LinAlgError:
            jitter = 10.0 * jitter if jitter > 0.0 else 1e-12
    raise np.linalg.LinAlgError("Failed to make covariance matrix positive definite.")


def _spd_inv_logdet(matrix: Array) -> tuple[Array, float]:
    spd = _ensure_spd(matrix)
    chol = np.linalg.cholesky(spd)
    inv = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(spd.shape[0], dtype=float)))
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return inv, logdet


def _normalise_rows(matrix: Array) -> Array:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _to_continuous_matrix(
    x: Sequence[Obs] | Array,
    *,
    name: str,
    expected_dim: Optional[int] = None,
) -> Array:
    if isinstance(x, np.ndarray):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(f"{name} expects a 1D/2D array, got shape {arr.shape}.")
    else:
        rows: list[list[float]] = []
        for idx, value in enumerate(x):
            if isinstance(value, bool):
                raise TypeError(f"{name} expects real-valued observations; got bool at index {idx}.")

            if isinstance(value, (int, float, np.integer, np.floating)):
                row = [float(value)]
            elif isinstance(value, np.ndarray):
                vec = np.asarray(value, dtype=float)
                if vec.ndim != 1 or vec.size == 0:
                    raise ValueError(
                        f"{name} expects 1D vectors for multivariate observations; got shape {vec.shape} at index {idx}."
                    )
                row = [float(v) for v in vec.tolist()]
            elif isinstance(value, (tuple, list)):
                if len(value) == 0:
                    raise ValueError(f"{name} received empty vector observation at index {idx}.")
                row = []
                for j, item in enumerate(value):
                    if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
                        raise TypeError(
                            f"{name} expects numeric vectors, got {type(item).__name__} at index {idx}:{j}."
                        )
                    row.append(float(item))
            else:
                raise TypeError(
                    f"{name} expects scalar floats/ints or float tuples, got {type(value).__name__} at index {idx}."
                )

            if not np.all(np.isfinite(row)):
                raise ValueError(f"{name} requires finite observations; got non-finite row at index {idx}.")
            rows.append(row)

        if not rows:
            if expected_dim is None:
                return np.zeros((0, 1), dtype=float)
            return np.zeros((0, expected_dim), dtype=float)

        width = len(rows[0])
        for idx, row in enumerate(rows[1:], start=1):
            if len(row) != width:
                raise ValueError(
                    f"{name} expects fixed-width observations; row 0 has width {width}, row {idx} has {len(row)}."
                )
        arr = np.asarray(rows, dtype=float)

    if arr.shape[0] == 0:
        if expected_dim is not None:
            return np.zeros((0, expected_dim), dtype=float)
        return arr

    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(
            f"{name} expected dimension {expected_dim}, got {arr.shape[1]}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} requires finite observations.")
    return arr


def _projection_pca(y: Array, macro_dim: int) -> Array:
    centered = y - np.mean(y, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    if macro_dim > vt.shape[0]:
        raise ValueError(
            f"macro_dim={macro_dim} exceeds available rank {vt.shape[0]} for PCA projection."
        )
    return _normalise_rows(vt[:macro_dim, :])


def _projection_random(obs_dim: int, macro_dim: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    return _normalise_rows(rng.normal(size=(macro_dim, obs_dim)))


def _compute_psi_with_backoff(
    innovations,
    projection: Array,
    *,
    tol: float,
    max_iter: int,
    ridge: float,
) -> float:
    last_error: Optional[Exception] = None
    for factor in (1.0, 10.0, 100.0, 1_000.0):
        try:
            return float(
                compute_ss_psi(
                    innovations,
                    projection,
                    tol=tol,
                    max_iter=max_iter,
                    ridge=ridge * factor,
                )
            )
        except Exception as exc:  # pragma: no cover - rare numeric fallback path
            last_error = exc
    raise RuntimeError("Failed to compute Psi after ridge backoff.") from last_error


@dataclass(frozen=True)
class MacroQuantiser:
    """Per-dimension bin edges for symbolising macro observations."""

    edges: tuple[Array, ...]
    radices: tuple[int, ...]

    def symbol(self, value: Array) -> int:
        digits: list[int] = []
        for idx, edge in enumerate(self.edges):
            digits.append(int(np.searchsorted(edge, float(value[idx]), side="right")))
        code = 0
        for digit, radix in zip(digits, self.radices):
            code = code * radix + digit
        return int(code)


def _build_quantiser(values: Array, bins: int, *, method: str) -> MacroQuantiser:
    if values.ndim != 2:
        raise ValueError(f"Quantiser expects 2D values, got shape {values.shape}.")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}.")
    if method != "quantile":
        raise ValueError(f"Unsupported macro symboliser method {method!r}.")
    q_points = np.linspace(0.0, 1.0, bins + 1, dtype=float)[1:-1]
    edges: list[Array] = []
    radices: list[int] = []
    for j in range(values.shape[1]):
        if q_points.size == 0:
            cuts = np.array([], dtype=float)
        else:
            cuts = np.quantile(values[:, j], q_points)
            cuts = np.unique(cuts.astype(float))
        edges.append(cuts)
        radices.append(int(cuts.size + 1))
    return MacroQuantiser(edges=tuple(edges), radices=tuple(radices))


def _symmetric_gaussian_kl(mu_a: Array, cov_a: Array, mu_b: Array, cov_b: Array) -> float:
    mu_a = np.asarray(mu_a, dtype=float).reshape(-1, 1)
    mu_b = np.asarray(mu_b, dtype=float).reshape(-1, 1)
    cov_a = _ensure_spd(cov_a)
    cov_b = _ensure_spd(cov_b)
    inv_a, logdet_a = _spd_inv_logdet(cov_a)
    inv_b, logdet_b = _spd_inv_logdet(cov_b)
    dim = mu_a.shape[0]
    delta_ab = mu_b - mu_a
    delta_ba = -delta_ab
    kl_ab = 0.5 * (
        (logdet_b - logdet_a)
        - dim
        + float(np.trace(inv_b @ cov_a))
        + float((delta_ab.T @ inv_b @ delta_ab).item())
    )
    kl_ba = 0.5 * (
        (logdet_a - logdet_b)
        - dim
        + float(np.trace(inv_a @ cov_b))
        + float((delta_ba.T @ inv_a @ delta_ba).item())
    )
    score = kl_ab + kl_ba
    if score < 0.0 and score > -1e-9:
        return 0.0
    return float(max(0.0, score))


def _reconstruct_macrostates(
    pred_means: Array,
    pred_covs: Array,
    *,
    eps: float,
) -> Array:
    if pred_means.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    order = sorted(
        range(pred_means.shape[0]),
        key=lambda idx: tuple(pred_means[idx].tolist()),
    )
    states = np.full(pred_means.shape[0], -1, dtype=int)
    current_state = 0
    current_bin: list[int] = []
    anchor_idx: Optional[int] = None

    def flush_bin() -> None:
        nonlocal current_state, current_bin
        for idx in current_bin:
            states[idx] = current_state
        current_state += 1
        current_bin = []

    for idx in order:
        if anchor_idx is None:
            anchor_idx = idx
            current_bin = [idx]
            continue
        distance = _symmetric_gaussian_kl(
            pred_means[anchor_idx],
            pred_covs[anchor_idx],
            pred_means[idx],
            pred_covs[idx],
        )
        if distance <= eps:
            current_bin.append(idx)
        else:
            flush_bin()
            anchor_idx = idx
            current_bin = [idx]

    if current_bin:
        flush_bin()

    if np.any(states < 0):
        raise RuntimeError("Continuous macrostate reconstruction failed to assign every index.")
    return states


def _build_macro_dynamics(
    y_train: Array,
    iss_model: KalmanISSModel,
    projection: Array,
    *,
    eps: float,
    macro_bins: int,
    macro_symboliser: str,
    steady_state: bool,
    steady_state_tol: float,
    steady_state_max_iter: int,
    steady_state_ridge: float,
    allow_time_varying_fallback: bool,
    steady_state_solution: SteadyStateKalmanSolution | None,
) -> tuple[dict[int, float], dict[tuple[int, int], dict[int, float]], dict[tuple[int, int], int], int]:
    if y_train.shape[0] < 3:
        raise ValueError("Need at least 3 training samples to build continuous macro transitions.")

    mu_y, s_y, _ = one_step_predictive_y(
        y_train,
        iss_model,
        steady_state=steady_state,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        steady_state_ridge=steady_state_ridge,
        steady_state_strict=not allow_time_varying_fallback,
        allow_time_varying_fallback=allow_time_varying_fallback,
        steady_state_solution=steady_state_solution,
    )

    n_pred = y_train.shape[0] - 1  # t = 0..T-2 predicts X_{t+1}
    macro_dim = projection.shape[0]
    pred_mu = np.zeros((n_pred, macro_dim), dtype=float)
    pred_cov = np.zeros((n_pred, macro_dim, macro_dim), dtype=float)

    for t in range(n_pred):
        mu_next = mu_y[t + 1]
        cov_next = s_y[t + 1]
        pred_mu[t] = (projection @ mu_next).reshape(-1)
        pred_cov[t] = _ensure_spd(projection @ cov_next @ projection.T)

    state_of_t = _reconstruct_macrostates(pred_mu, pred_cov, eps=eps)
    occupancy = Counter(int(s) for s in state_of_t.tolist())
    occ_total = sum(occupancy.values()) or 1
    pi = {state: count / occ_total for state, count in occupancy.items()}

    macro_obs = y_train @ projection.T  # shape (T, d_V)
    quantiser = _build_quantiser(macro_obs[1:], bins=macro_bins, method=macro_symboliser)

    trans_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    sa_counts: dict[tuple[int, int], int] = {}
    for t in range(n_pred - 1):
        s_t = int(state_of_t[t])
        s_tp1 = int(state_of_t[t + 1])
        symbol = quantiser.symbol(macro_obs[t + 1])
        trans_counts[(s_t, symbol)][s_tp1] += 1

    transitions: dict[tuple[int, int], dict[int, float]] = {}
    for key, ctr in trans_counts.items():
        denom = sum(ctr.values())
        if denom <= 0:
            continue
        transitions[key] = {sp: c / denom for sp, c in ctr.items()}
        sa_counts[key] = int(denom)

    return pi, transitions, sa_counts, len(set(state_of_t.tolist()))


@dataclass(frozen=True)
class GaussianPredictiveStateModel:
    iss: KalmanISSModel
    latent_dim: int
    obs_dim: int
    macro_dim: int
    n_macro_states: int
    train_avg_nll: float
    pi: dict[int, float]
    transitions: dict[tuple[int, int], dict[int, float]]
    sa_counts: dict[tuple[int, int], int]
    projection: Array
    projection_mode: str
    macro_eps: float
    macro_bins: int
    macro_symboliser: str
    iss_mode: str
    allow_time_varying_fallback: bool
    steady_state_tol: float
    steady_state_max_iter: int
    steady_state_ridge: float
    steady_state_solution: Optional[SteadyStateKalmanSolution]
    psi_opt: float
    psi_macro_dim: int
    psi_optimiser: str
    psi_restarts: int
    psi_iterations: int
    psi_L: Array

    def _steady_state_enabled(self) -> bool:
        return self.iss_mode == "steady_state"

    def predictive_distributions(
        self,
        observations: Sequence[Obs] | Array,
        *,
        context: Optional[Sequence[Obs] | Array] = None,
    ) -> Tuple[Array, Array]:
        y_obs = _to_continuous_matrix(
            observations,
            name="GaussianPredictiveStateModel.predictive_distributions",
            expected_dim=self.obs_dim,
        )
        if y_obs.shape[0] == 0:
            return (
                np.zeros((0, self.obs_dim, 1), dtype=float),
                np.zeros((0, self.obs_dim, self.obs_dim), dtype=float),
            )
        if context is not None:
            y_ctx = _to_continuous_matrix(
                context,
                name="GaussianPredictiveStateModel.predictive_distributions",
                expected_dim=self.obs_dim,
            )
            if y_ctx.shape[0] > 0:
                y_all = np.vstack([y_ctx, y_obs])
                mu_all, s_all, _ = one_step_predictive_y(
                    y_all,
                    self.iss,
                    steady_state=self._steady_state_enabled(),
                    steady_state_tol=self.steady_state_tol,
                    steady_state_max_iter=self.steady_state_max_iter,
                    steady_state_ridge=self.steady_state_ridge,
                    steady_state_strict=not self.allow_time_varying_fallback,
                    allow_time_varying_fallback=self.allow_time_varying_fallback,
                    steady_state_solution=self.steady_state_solution,
                )
                start = y_ctx.shape[0]
                return mu_all[start:], s_all[start:]
        mu_obs, s_obs, _ = one_step_predictive_y(
            y_obs,
            self.iss,
            steady_state=self._steady_state_enabled(),
            steady_state_tol=self.steady_state_tol,
            steady_state_max_iter=self.steady_state_max_iter,
            steady_state_ridge=self.steady_state_ridge,
            steady_state_strict=not self.allow_time_varying_fallback,
            allow_time_varying_fallback=self.allow_time_varying_fallback,
            steady_state_solution=self.steady_state_solution,
        )
        return mu_obs, s_obs

    def average_negative_log_likelihood(
        self,
        observations: Sequence[Obs] | Array,
        *,
        context: Optional[Sequence[Obs] | Array] = None,
    ) -> float:
        y_obs = _to_continuous_matrix(
            observations,
            name="GaussianPredictiveStateModel.average_negative_log_likelihood",
            expected_dim=self.obs_dim,
        )
        if y_obs.shape[0] == 0:
            return math.nan
        mu_obs, s_obs = self.predictive_distributions(observations, context=context)
        losses = [_gaussian_nll(y_obs[t], mu_obs[t], s_obs[t]) for t in range(y_obs.shape[0])]
        return float(sum(losses) / len(losses))

    def filtered_state_sequence(
        self,
        observations: Sequence[Obs] | Array,
        *,
        context: Optional[Sequence[Obs] | Array] = None,
    ) -> Tuple[Array, Array]:
        y_obs = _to_continuous_matrix(
            observations,
            name="GaussianPredictiveStateModel.filtered_state_sequence",
            expected_dim=self.obs_dim,
        )
        if y_obs.shape[0] == 0:
            return (
                np.zeros((0, self.latent_dim, 1), dtype=float),
                np.zeros((0, self.latent_dim, self.latent_dim), dtype=float),
            )
        if context is not None:
            y_ctx = _to_continuous_matrix(
                context,
                name="GaussianPredictiveStateModel.filtered_state_sequence",
                expected_dim=self.obs_dim,
            )
            if y_ctx.shape[0] > 0:
                y_all = np.vstack([y_ctx, y_obs])
                mu_f, p_f, _, _, _ = iss_filter(
                    y_all,
                    self.iss,
                    steady_state=self._steady_state_enabled(),
                    steady_state_tol=self.steady_state_tol,
                    steady_state_max_iter=self.steady_state_max_iter,
                    steady_state_ridge=self.steady_state_ridge,
                    steady_state_strict=not self.allow_time_varying_fallback,
                    allow_time_varying_fallback=self.allow_time_varying_fallback,
                    steady_state_solution=self.steady_state_solution,
                )
                start = y_ctx.shape[0]
                return mu_f[start:], p_f[start:]
        mu_f, p_f, _, _, _ = iss_filter(
            y_obs,
            self.iss,
            steady_state=self._steady_state_enabled(),
            steady_state_tol=self.steady_state_tol,
            steady_state_max_iter=self.steady_state_max_iter,
            steady_state_ridge=self.steady_state_ridge,
            steady_state_strict=not self.allow_time_varying_fallback,
            allow_time_varying_fallback=self.allow_time_varying_fallback,
            steady_state_solution=self.steady_state_solution,
        )
        return mu_f, p_f


@dataclass
class KalmanISSReconstructor(Reconstructor[GaussianPredictiveStateModel]):
    em_iters: int = 50
    em_tol: float = 1e-4
    em_ridge: float = 1e-6
    min_train_samples: int = 30
    macro_eps: float = 0.25
    macro_bins: int = 3
    macro_symboliser: str = "quantile"
    projection_mode: str = "pca"  # pca | random | psi_opt
    iss_mode: str = "steady_state"  # steady_state | time_varying
    allow_time_varying_fallback: bool = False
    steady_state_tol: float = 1e-9
    steady_state_max_iter: int = 10_000
    steady_state_ridge: float = 1e-9
    compute_psi: bool = False
    psi_optimiser: str = "random"
    psi_restarts: int = 12
    psi_iterations: int = 120
    psi_lr: float = 0.03
    psi_step_scale: float = 0.2
    psi_tol: float = 1e-8
    psi_max_iter: int = 4_000
    psi_ridge: float = 1e-8

    @property
    def name(self) -> str:
        return "kalman_iss"

    @property
    def eps(self) -> float:
        # Kept for backwards-compatible CSV schema.
        return float(self.macro_eps)

    def _projection(self, y_train: Array, macro_dim: int, seed: int) -> tuple[Array, str]:
        mode = self.projection_mode
        if mode == "pca":
            return _projection_pca(y_train, macro_dim), "pca"
        if mode == "random":
            return _projection_random(y_train.shape[1], macro_dim, seed=seed), "random"
        if mode == "psi_opt":
            # placeholder projection; caller replaces with optimised L.
            return _projection_pca(y_train, macro_dim), "psi_opt"
        raise ValueError(f"Unknown projection_mode={mode!r}. Expected 'pca', 'random', or 'psi_opt'.")

    def fit(
        self,
        x_train: Sequence[Obs],
        rep: Representation,
        seed: int = 0,
    ) -> GaussianPredictiveStateModel:
        if not isinstance(rep, ISSDim):
            raise TypeError(
                f"KalmanISSReconstructor expects ISSDim representation, got {type(rep).__name__}."
            )
        latent_dim = rep.d
        macro_dim = rep.dv

        y_train = _to_continuous_matrix(x_train, name="KalmanISSReconstructor.fit")
        if y_train.shape[1] < macro_dim:
            raise ValueError(
                f"macro dimension dv={macro_dim} exceeds observation dimension p={y_train.shape[1]}."
            )
        if y_train.shape[0] < max(self.min_train_samples, 5 * latent_dim):
            raise ValueError(
                f"Need at least {max(self.min_train_samples, 5 * latent_dim)} training samples for "
                f"stable ISS fitting with latent_dim={latent_dim}; got {y_train.shape[0]}."
            )
        if self.macro_eps < 0.0:
            raise ValueError("macro_eps must be >= 0.")
        if self.macro_bins < 1:
            raise ValueError("macro_bins must be >= 1.")
        if self.macro_symboliser != "quantile":
            raise ValueError("macro_symboliser must be 'quantile'.")
        if self.iss_mode not in {"steady_state", "time_varying"}:
            raise ValueError("iss_mode must be 'steady_state' or 'time_varying'.")
        if self.steady_state_tol <= 0.0:
            raise ValueError("steady_state_tol must be > 0.")
        if self.steady_state_max_iter < 1:
            raise ValueError("steady_state_max_iter must be >= 1.")
        if self.steady_state_ridge < 0.0:
            raise ValueError("steady_state_ridge must be >= 0.")

        cfg = KalmanISSConfig(
            latent_dim=latent_dim,
            em_iters=self.em_iters,
            tol=self.em_tol,
            ridge=self.em_ridge,
            seed=seed,
        )
        iss_model = fit_kalman_iss_em(y_train, cfg)
        steady_state_enabled = self.iss_mode == "steady_state"
        steady_state_solution: Optional[SteadyStateKalmanSolution] = None
        if steady_state_enabled:
            steady_state_solution = solve_steady_state_kalman(
                iss_model,
                tol=self.steady_state_tol,
                max_iter=self.steady_state_max_iter,
                ridge=self.steady_state_ridge,
                strict=not self.allow_time_varying_fallback,
            )
            if not steady_state_solution.converged and self.allow_time_varying_fallback:
                steady_state_enabled = False
                steady_state_solution = None

        mu_train, s_train, _ = one_step_predictive_y(
            y_train,
            iss_model,
            steady_state=steady_state_enabled,
            steady_state_tol=self.steady_state_tol,
            steady_state_max_iter=self.steady_state_max_iter,
            steady_state_ridge=self.steady_state_ridge,
            steady_state_strict=not self.allow_time_varying_fallback,
            allow_time_varying_fallback=self.allow_time_varying_fallback,
            steady_state_solution=steady_state_solution,
        )
        train_losses = [_gaussian_nll(y_train[t], mu_train[t], s_train[t]) for t in range(y_train.shape[0])]
        avg_train_nll = float(sum(train_losses) / len(train_losses))

        projection, projection_label = self._projection(y_train, macro_dim=macro_dim, seed=seed)
        innovations = innovations_from_ssm(
            iss_model,
            tol=self.psi_tol,
            max_iter=self.psi_max_iter,
            ridge=self.psi_ridge,
            strict=not self.allow_time_varying_fallback,
        )
        psi_opt = _compute_psi_with_backoff(
            innovations,
            projection,
            tol=self.psi_tol,
            max_iter=self.psi_max_iter,
            ridge=self.psi_ridge,
        )
        psi_optimiser = projection_label
        psi_restarts = 1
        psi_iterations = 1
        psi_L = projection.copy()

        should_optimise_psi = self.compute_psi or self.projection_mode == "psi_opt"
        if should_optimise_psi:
            try:
                psi_result = optimise_ss_psi(
                    innovations,
                    macro_dim=macro_dim,
                    seed=seed,
                    optimiser=self.psi_optimiser,
                    restarts=self.psi_restarts,
                    iterations=self.psi_iterations,
                    lr=self.psi_lr,
                    step_scale=self.psi_step_scale,
                    tol=self.psi_tol,
                    max_iter=self.psi_max_iter,
                    ridge=self.psi_ridge,
                )
                projection = psi_result.L
                psi_opt = float(psi_result.psi)
                psi_optimiser = psi_result.optimiser
                psi_restarts = psi_result.restarts
                psi_iterations = psi_result.iterations
                psi_L = psi_result.L.copy()
            except Exception:  # pragma: no cover - rare numeric fallback path
                psi_opt = _compute_psi_with_backoff(
                    innovations,
                    projection,
                    tol=self.psi_tol,
                    max_iter=self.psi_max_iter,
                    ridge=self.psi_ridge,
                )
                psi_optimiser = f"{projection_label}_fallback"
                psi_restarts = 1
                psi_iterations = 1
                psi_L = projection.copy()

        pi, transitions, sa_counts, n_macro_states = _build_macro_dynamics(
            y_train=y_train,
            iss_model=iss_model,
            projection=projection,
            eps=self.macro_eps,
            macro_bins=self.macro_bins,
            macro_symboliser=self.macro_symboliser,
            steady_state=steady_state_enabled,
            steady_state_tol=self.steady_state_tol,
            steady_state_max_iter=self.steady_state_max_iter,
            steady_state_ridge=self.steady_state_ridge,
            allow_time_varying_fallback=self.allow_time_varying_fallback,
            steady_state_solution=steady_state_solution,
        )

        return GaussianPredictiveStateModel(
            iss=iss_model,
            latent_dim=latent_dim,
            obs_dim=y_train.shape[1],
            macro_dim=macro_dim,
            n_macro_states=n_macro_states,
            train_avg_nll=avg_train_nll,
            pi=pi,
            transitions=transitions,
            sa_counts=sa_counts,
            projection=projection,
            projection_mode=projection_label,
            macro_eps=self.macro_eps,
            macro_bins=self.macro_bins,
            macro_symboliser=self.macro_symboliser,
            iss_mode="steady_state" if steady_state_enabled else "time_varying",
            allow_time_varying_fallback=self.allow_time_varying_fallback,
            steady_state_tol=self.steady_state_tol,
            steady_state_max_iter=self.steady_state_max_iter,
            steady_state_ridge=self.steady_state_ridge,
            steady_state_solution=steady_state_solution,
            psi_opt=float(psi_opt),
            psi_macro_dim=macro_dim,
            psi_optimiser=psi_optimiser,
            psi_restarts=psi_restarts,
            psi_iterations=psi_iterations,
            psi_L=psi_L,
        )
