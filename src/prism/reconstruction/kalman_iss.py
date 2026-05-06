from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
import time
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


def _average_gaussian_nll(y: Array, means: Array, covariances: Array) -> float:
    if y.shape[0] == 0:
        return math.nan

    if covariances.shape[0] == y.shape[0] and np.all(covariances == covariances[0]):
        cov = _ensure_spd(covariances[0])
        inv, logdet = _spd_inv_logdet(cov)
        dim = int(y.shape[1])
        const = dim * np.log(2.0 * np.pi)
        losses: list[float] = []
        for t in range(y.shape[0]):
            diff = (
                np.asarray(y[t], dtype=float).reshape(-1, 1)
                - np.asarray(means[t], dtype=float).reshape(-1, 1)
            )
            quad = float((diff.T @ inv @ diff).item())
            losses.append(0.5 * (const + logdet + quad))
        return float(sum(losses) / len(losses))

    losses = [_gaussian_nll(y[t], means[t], covariances[t]) for t in range(y.shape[0])]
    return float(sum(losses) / len(losses))


def _stack_rows(first: Array, second: Array) -> Array:
    out = np.empty(
        (first.shape[0] + second.shape[0], first.shape[1]),
        dtype=np.result_type(first.dtype, second.dtype),
    )
    out[: first.shape[0]] = first
    out[first.shape[0] :] = second
    return out


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


def _reconstruct_macrostates_greedy(
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


def _pairwise_sym_kl(pred_means: Array, pred_covs: Array) -> Array:
    n = pred_means.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    if np.all(pred_covs == pred_covs[0]):
        cov = _ensure_spd(pred_covs[0])
        inv, _ = _spd_inv_logdet(cov)
        dim = int(pred_means.shape[1])
        self_term = float(np.trace(inv @ cov)) - dim
        gram = pred_means @ inv @ pred_means.T
        diag = np.diag(gram)
        dist = diag[:, None] + diag[None, :] - 2.0 * gram + self_term
        np.maximum(dist, 0.0, out=dist)
        np.fill_diagonal(dist, 0.0)
        return dist

    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            val = _symmetric_gaussian_kl(pred_means[i], pred_covs[i], pred_means[j], pred_covs[j])
            dist[i, j] = val
            dist[j, i] = val
    return dist


def _reconstruct_macrostates_global_singlelink(dist: Array, *, eps: float) -> Array:
    n = dist.shape[0]
    labels = np.full(n, -1, dtype=int)
    cluster = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = cluster
        stack = [i]
        while stack:
            u = stack.pop()
            neighbours = np.where((dist[u] <= eps) & (labels < 0))[0]
            for v in neighbours.tolist():
                labels[v] = cluster
                stack.append(int(v))
        cluster += 1
    return labels


def _reconstruct_macrostates_global_complete(dist: Array, *, eps: float) -> Array:
    n = dist.shape[0]
    clusters: list[list[int]] = [[i] for i in range(n)]
    cluster_dist = dist.copy()
    np.fill_diagonal(cluster_dist, math.inf)
    active_ids = list(range(n))

    while True:
        if len(active_ids) < 2:
            break

        active_dist = cluster_dist[np.ix_(active_ids, active_ids)].copy()
        np.fill_diagonal(active_dist, math.inf)
        flat_idx = int(np.argmin(active_dist))
        best_cost = float(active_dist.flat[flat_idx])
        if not math.isfinite(best_cost) or best_cost > eps:
            break

        pos_i, pos_j = divmod(flat_idx, len(active_ids))
        if pos_i > pos_j:
            pos_i, pos_j = pos_j, pos_i
        best_i = active_ids[pos_i]
        best_j = active_ids[pos_j]

        clusters[best_i].extend(clusters[best_j])
        del active_ids[pos_j]

        # Complete-linkage distance from the merged cluster to any active
        # cluster is the maximum of the two previous cluster distances.
        if active_ids:
            active = np.asarray(active_ids, dtype=int)
            updated = np.maximum(cluster_dist[best_i, active], cluster_dist[best_j, active])
            cluster_dist[best_i, active] = updated
            cluster_dist[active, best_i] = updated
        cluster_dist[best_i, best_i] = math.inf
        cluster_dist[best_j, :] = math.inf
        cluster_dist[:, best_j] = math.inf

    labels = np.full(n, -1, dtype=int)
    for idx, group_id in enumerate(active_ids):
        group = clusters[group_id]
        for pos in group:
            labels[pos] = idx
    return labels


def _normalise_macro_builder(builder: str) -> str:
    aliases = {
        "global": "hierarchical_complete",
        "global_complete": "hierarchical_complete",
        "global_single": "hierarchical_single",
        "linear": "linear_quantile",
        "linear_baseline": "linear_quantile",
    }
    return aliases.get(builder, builder)


def _valid_macro_builder(builder: str) -> bool:
    mode = _normalise_macro_builder(builder)
    return mode in {"greedy", "hierarchical_single", "hierarchical_complete", "linear_quantile"}


def _reconstruct_macrostates(
    pred_means: Array,
    pred_covs: Array,
    *,
    eps: float,
    builder: str,
) -> Array:
    mode = _normalise_macro_builder(builder)
    if mode == "greedy":
        return _reconstruct_macrostates_greedy(pred_means, pred_covs, eps=eps)
    if mode == "linear_quantile":
        raise ValueError("linear_quantile does not use predictor clustering.")
    dist = _pairwise_sym_kl(pred_means, pred_covs)
    if mode == "hierarchical_complete":
        return _reconstruct_macrostates_global_complete(dist, eps=eps)
    if mode == "hierarchical_single":
        return _reconstruct_macrostates_global_singlelink(dist, eps=eps)
    raise ValueError(
        f"Unknown macro_builder={builder!r}. Expected 'greedy', 'hierarchical_single', 'hierarchical_complete', or 'linear_quantile'."
    )


def _state_prototypes(pred_mu: Array, pred_cov: Array, labels: Array) -> tuple[Array, Array]:
    n_states = int(np.max(labels)) + 1
    macro_dim = pred_mu.shape[1]
    centres_mu = np.zeros((n_states, macro_dim), dtype=float)
    centres_cov = np.zeros((n_states, macro_dim, macro_dim), dtype=float)
    for state in range(n_states):
        idx = np.where(labels == state)[0]
        if idx.size == 0:
            raise RuntimeError(f"State {state} has no support.")
        mu_s = np.mean(pred_mu[idx], axis=0)
        centred = pred_mu[idx] - mu_s[None, :]
        between = (centred.T @ centred) / max(idx.size, 1)
        within = np.mean(pred_cov[idx], axis=0)
        centres_mu[state] = mu_s
        centres_cov[state] = _ensure_spd(within + between)
    return centres_mu, centres_cov


def _assign_states(pred_mu: Array, pred_cov: Array, centres_mu: Array, centres_cov: Array) -> Array:
    n = pred_mu.shape[0]
    out = np.zeros(n, dtype=int)
    for t in range(n):
        best_state = 0
        best_dist = math.inf
        for s in range(centres_mu.shape[0]):
            val = _symmetric_gaussian_kl(pred_mu[t], pred_cov[t], centres_mu[s], centres_cov[s])
            if val < best_dist:
                best_dist = val
                best_state = s
        out[t] = best_state
    return out


def _compact_labels(labels: Array) -> Array:
    uniques = sorted({int(v) for v in labels.tolist()})
    remap = {state: idx for idx, state in enumerate(uniques)}
    return np.asarray([remap[int(v)] for v in labels.tolist()], dtype=int)


def _transition_diagnostics(
    transitions: dict[tuple[int, int], dict[int, float]],
    sa_counts: dict[tuple[int, int], int],
) -> tuple[float, float]:
    if not transitions:
        return math.nan, math.nan
    total_w = 0.0
    unif_total = 0.0
    branch_total = 0.0
    for key, dist in transitions.items():
        if not dist:
            continue
        w = float(sa_counts.get(key, 0))
        if w <= 0.0:
            continue
        local_unif = max(dist.values())
        local_branch = 0.0
        for p in dist.values():
            if p > 0.0:
                local_branch -= p * math.log(p, 2)
        total_w += w
        unif_total += w * local_unif
        branch_total += w * local_branch
    if total_w <= 0.0:
        return math.nan, math.nan
    return float(unif_total / total_w), float(branch_total / total_w)


def _variance(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return 0.0
    mean_val = float(sum(clean) / len(clean))
    return float(sum((v - mean_val) ** 2 for v in clean) / (len(clean) - 1))


def _mean(values: Sequence[float], *, default: float = math.nan) -> float:
    clean = [float(v) for v in values if math.isfinite(v)]
    if not clean:
        return default
    return float(sum(clean) / len(clean))


def _max(values: Sequence[float], *, default: float = math.nan) -> float:
    clean = [float(v) for v in values if math.isfinite(v)]
    if not clean:
        return default
    return float(max(clean))


def _distance_memory_mb(builder: str, n_pred: int) -> float:
    mode = _normalise_macro_builder(builder)
    if mode not in {"hierarchical_single", "hierarchical_complete"}:
        return 0.0
    return float((n_pred * n_pred * 8.0) / (1024.0 * 1024.0))


def _blocked_resample(y: Array, *, frac: float, min_len: int, seed: int) -> Array:
    if y.shape[0] <= min_len:
        return y
    span = int(round(y.shape[0] * frac))
    span = max(min_len, min(span, y.shape[0]))
    if span >= y.shape[0]:
        return y
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, y.shape[0] - span + 1))
    return y[start : start + span]


@dataclass(frozen=True)
class MacroBuild:
    pi: dict[int, float]
    transitions: dict[tuple[int, int], dict[int, float]]
    sa_counts: dict[tuple[int, int], int]
    n_macro_states: int
    labels: Array
    pred_mu: Array
    pred_cov: Array
    macro_obs: Array
    quantiser: MacroQuantiser
    centres_mu: Array
    centres_cov: Array
    unifilarity: float
    branch_entropy: float
    build_time_s: float
    distance_matrix_mb_est: float
    n_pred: int


def _build_macro_dynamics(
    y_train: Array,
    iss_model: KalmanISSModel,
    projection: Array,
    *,
    eps: float,
    macro_bins: int,
    macro_symboliser: str,
    macro_builder: str,
    steady_state: bool,
    steady_state_tol: float,
    steady_state_max_iter: int,
    steady_state_ridge: float,
    allow_time_varying_fallback: bool,
    steady_state_solution: SteadyStateKalmanSolution | None,
) -> MacroBuild:
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

    cov_nexts = s_y[1 : n_pred + 1]
    if np.all(cov_nexts == cov_nexts[0]):
        projected_cov = _ensure_spd(projection @ cov_nexts[0] @ projection.T)
        for t in range(n_pred):
            pred_mu[t] = (projection @ mu_y[t + 1]).reshape(-1)
        pred_cov[:] = projected_cov
    else:
        for t in range(n_pred):
            mu_next = mu_y[t + 1]
            cov_next = s_y[t + 1]
            pred_mu[t] = (projection @ mu_next).reshape(-1)
            pred_cov[t] = _ensure_spd(projection @ cov_next @ projection.T)

    macro_obs = y_train @ projection.T  # shape (T, d_V)
    quantiser = _build_quantiser(macro_obs[1:], bins=macro_bins, method=macro_symboliser)

    t0 = time.perf_counter()
    mode = _normalise_macro_builder(macro_builder)
    if mode == "linear_quantile":
        state_of_t = np.zeros(n_pred, dtype=int)
        for t in range(n_pred):
            state_of_t[t] = quantiser.symbol(macro_obs[t + 1])
    else:
        state_of_t = _reconstruct_macrostates(pred_mu, pred_cov, eps=eps, builder=macro_builder)
    state_of_t = _compact_labels(state_of_t)
    build_time = float(time.perf_counter() - t0)
    mem_mb_est = _distance_memory_mb(macro_builder, n_pred)

    occupancy = Counter(int(s) for s in state_of_t.tolist())
    occ_total = sum(occupancy.values()) or 1
    pi = {state: count / occ_total for state, count in occupancy.items()}

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

    centres_mu, centres_cov = _state_prototypes(pred_mu, pred_cov, state_of_t)
    unif, branch = _transition_diagnostics(transitions, sa_counts)
    return MacroBuild(
        pi=pi,
        transitions=transitions,
        sa_counts=sa_counts,
        n_macro_states=len(set(state_of_t.tolist())),
        labels=state_of_t,
        pred_mu=pred_mu,
        pred_cov=pred_cov,
        macro_obs=macro_obs,
        quantiser=quantiser,
        centres_mu=centres_mu,
        centres_cov=centres_cov,
        unifilarity=unif,
        branch_entropy=branch,
        build_time_s=build_time,
        distance_matrix_mb_est=mem_mb_est,
        n_pred=n_pred,
    )


@dataclass(frozen=True)
class CandidateSpec:
    projection_mode: str
    macro_builder: str
    macro_eps: float


@dataclass(frozen=True)
class CandidateScore:
    spec: CandidateSpec
    predictive: float
    stability: float
    chosen: float
    n_states_var: float
    unifilarity_var: float
    branch_entropy_var: float
    n_states_mean: float
    unifilarity_mean: float
    branch_entropy_mean: float
    macro_time_mean_s: float
    macro_time_max_s: float
    distance_matrix_mb_est: float
    n_pred_mean: float


def _project_predictors(mu_y: Array, s_y: Array, projection: Array, t_start: int, t_stop: int) -> tuple[Array, Array]:
    if t_stop < t_start:
        return (
            np.zeros((0, projection.shape[0]), dtype=float),
            np.zeros((0, projection.shape[0], projection.shape[0]), dtype=float),
        )
    n = t_stop - t_start + 1
    out_mu = np.zeros((n, projection.shape[0]), dtype=float)
    out_cov = np.zeros((n, projection.shape[0], projection.shape[0]), dtype=float)
    cov_window = s_y[t_start + 1 : t_stop + 2]
    if np.all(cov_window == cov_window[0]):
        projected_cov = _ensure_spd(projection @ cov_window[0] @ projection.T)
        for i, t in enumerate(range(t_start, t_stop + 1)):
            out_mu[i] = (projection @ mu_y[t + 1]).reshape(-1)
        out_cov[:] = projected_cov
    else:
        for i, t in enumerate(range(t_start, t_stop + 1)):
            out_mu[i] = (projection @ mu_y[t + 1]).reshape(-1)
            out_cov[i] = _ensure_spd(projection @ s_y[t + 1] @ projection.T)
    return out_mu, out_cov


def _macro_validation_nll(
    y_fit: Array,
    y_val: Array,
    iss_model: KalmanISSModel,
    projection: Array,
    macro: MacroBuild,
    *,
    steady_state: bool,
    steady_state_tol: float,
    steady_state_max_iter: int,
    steady_state_ridge: float,
    allow_time_varying_fallback: bool,
    steady_state_solution: SteadyStateKalmanSolution | None,
) -> float:
    if y_val.shape[0] < 2:
        return math.nan
    y_all = _stack_rows(y_fit, y_val)
    mu_y, s_y, _ = one_step_predictive_y(
        y_all,
        iss_model,
        steady_state=steady_state,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        steady_state_ridge=steady_state_ridge,
        steady_state_strict=not allow_time_varying_fallback,
        allow_time_varying_fallback=allow_time_varying_fallback,
        steady_state_solution=steady_state_solution,
    )
    fit_len = y_fit.shape[0]
    val_len = y_val.shape[0]
    t0 = fit_len - 1
    t1 = fit_len + val_len - 2
    pred_mu_val, pred_cov_val = _project_predictors(mu_y, s_y, projection, t0, t1)
    labels_val = _assign_states(pred_mu_val, pred_cov_val, macro.centres_mu, macro.centres_cov)
    macro_obs_all = y_all @ projection.T

    total = 0.0
    steps = 0
    floor = 1e-12
    for j in range(labels_val.shape[0] - 1):
        t = t0 + j
        s_t = int(labels_val[j])
        s_tp1 = int(labels_val[j + 1])
        symbol = macro.quantiser.symbol(macro_obs_all[t + 1])
        row = macro.transitions.get((s_t, symbol))
        if row is None:
            prob = 1.0 / max(macro.n_macro_states, 1)
        else:
            prob = float(row.get(s_tp1, 0.0))
            if prob <= 0.0:
                prob = floor
        total += -math.log(max(prob, floor))
        steps += 1
    if steps == 0:
        return math.nan
    return float(total / steps)


def _comb2(n: int) -> float:
    if n < 2:
        return 0.0
    return float(n * (n - 1) // 2)


def _adjusted_rand_index(labels_a: Array, labels_b: Array) -> float:
    if labels_a.shape != labels_b.shape:
        raise ValueError("ARI expects equally sized label arrays.")
    n = int(labels_a.shape[0])
    if n < 2:
        return 1.0

    table: dict[tuple[int, int], int] = defaultdict(int)
    a_totals: dict[int, int] = defaultdict(int)
    b_totals: dict[int, int] = defaultdict(int)
    for a, b in zip(labels_a.tolist(), labels_b.tolist()):
        ai = int(a)
        bi = int(b)
        table[(ai, bi)] += 1
        a_totals[ai] += 1
        b_totals[bi] += 1

    index = sum(_comb2(v) for v in table.values())
    a_sum = sum(_comb2(v) for v in a_totals.values())
    b_sum = sum(_comb2(v) for v in b_totals.values())
    pairs = _comb2(n)
    if pairs <= 0.0:
        return 1.0
    expected = (a_sum * b_sum) / pairs
    max_index = 0.5 * (a_sum + b_sum)
    denom = max_index - expected
    if abs(denom) <= 1e-12:
        return 1.0
    return float((index - expected) / denom)


def _mean_pairwise_ari(label_sets: Sequence[Array]) -> float:
    if len(label_sets) < 2:
        return 1.0
    vals: list[float] = []
    for i, j in combinations(range(len(label_sets)), 2):
        vals.append(_adjusted_rand_index(label_sets[i], label_sets[j]))
    if not vals:
        return 1.0
    return float(sum(vals) / len(vals))


def _blocked_inner_split(y: Array, *, frac: float, min_fit: int) -> tuple[Array, Array]:
    if not (0.55 <= frac < 1.0):
        raise ValueError(f"selection_train_frac must lie in [0.55, 1.0), got {frac}.")
    n = y.shape[0]
    split = int(n * frac)
    split = max(split, min_fit)
    split = min(split, n - 2)
    if split < min_fit or (n - split) < 2:
        raise ValueError("Not enough samples for inner blocked train/validation split.")
    return y[:split], y[split:]


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
    macro_builder: str
    macro_eps: float
    macro_bins: int
    macro_symboliser: str
    model_selection: bool
    selection_score: str
    selection_value: float
    selection_predictive: float
    selection_stability: float
    selection_n_states_var: float
    selection_unifilarity_var: float
    selection_branch_entropy_var: float
    selection_n_states_mean: float
    selection_unifilarity_mean: float
    selection_branch_entropy_mean: float
    selection_macro_time_mean_s: float
    selection_macro_time_max_s: float
    selection_distance_matrix_mb_est: float
    selection_n_pred: float
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
    macro_build_time_s: float
    macro_distance_matrix_mb_est: float
    macro_n_pred: int

    def _steady_state_enabled(self) -> bool:
        return self.iss_mode == "steady_state"

    def macro_labels_at_eps(
        self,
        y_train: Array,
        eps: float,
    ) -> Array:
        """Return macro labels for one training sequence at a chosen epsilon."""
        macro = _build_macro_dynamics(
            y_train=y_train,
            iss_model=self.iss,
            projection=self.projection,
            eps=eps,
            macro_bins=self.macro_bins,
            macro_symboliser=self.macro_symboliser,
            macro_builder=self.macro_builder,
            steady_state=self._steady_state_enabled(),
            steady_state_tol=self.steady_state_tol,
            steady_state_max_iter=self.steady_state_max_iter,
            steady_state_ridge=self.steady_state_ridge,
            allow_time_varying_fallback=self.allow_time_varying_fallback,
            steady_state_solution=self.steady_state_solution,
        )
        return macro.labels

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
                y_all = _stack_rows(y_ctx, y_obs)
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
        return _average_gaussian_nll(y_obs, mu_obs, s_obs)

    def macro_average_negative_log_likelihood(
        self,
        observations: Sequence[Obs] | Array,
        *,
        context: Optional[Sequence[Obs] | Array] = None,
    ) -> float:
        if context is None:
            return math.nan
        y_fit = _to_continuous_matrix(
            context,
            name="GaussianPredictiveStateModel.macro_average_negative_log_likelihood",
            expected_dim=self.obs_dim,
        )
        y_obs = _to_continuous_matrix(
            observations,
            name="GaussianPredictiveStateModel.macro_average_negative_log_likelihood",
            expected_dim=self.obs_dim,
        )
        if y_fit.shape[0] < 3 or y_obs.shape[0] < 2:
            return math.nan

        macro = _build_macro_dynamics(
            y_train=y_fit,
            iss_model=self.iss,
            projection=self.projection,
            eps=self.macro_eps,
            macro_bins=self.macro_bins,
            macro_symboliser=self.macro_symboliser,
            macro_builder=self.macro_builder,
            steady_state=self._steady_state_enabled(),
            steady_state_tol=self.steady_state_tol,
            steady_state_max_iter=self.steady_state_max_iter,
            steady_state_ridge=self.steady_state_ridge,
            allow_time_varying_fallback=self.allow_time_varying_fallback,
            steady_state_solution=self.steady_state_solution,
        )
        return _macro_validation_nll(
            y_fit,
            y_obs,
            self.iss,
            self.projection,
            macro,
            steady_state=self._steady_state_enabled(),
            steady_state_tol=self.steady_state_tol,
            steady_state_max_iter=self.steady_state_max_iter,
            steady_state_ridge=self.steady_state_ridge,
            allow_time_varying_fallback=self.allow_time_varying_fallback,
            steady_state_solution=self.steady_state_solution,
        )

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
                y_all = _stack_rows(y_ctx, y_obs)
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

    def macro_label_sequence(
        self,
        y_train: Array,
        *,
        eps: float | None = None,
        builder: str | None = None,
    ) -> Array:
        """Rebuild one macro label sequence from the stored ISS fit."""
        macro = _build_macro_dynamics(
            y_train=np.asarray(y_train, dtype=float),
            iss_model=self.iss,
            projection=self.projection,
            eps=self.macro_eps if eps is None else float(eps),
            macro_bins=self.macro_bins,
            macro_symboliser=self.macro_symboliser,
            macro_builder=self.macro_builder if builder is None else str(builder),
            steady_state=self._steady_state_enabled(),
            steady_state_tol=self.steady_state_tol,
            steady_state_max_iter=self.steady_state_max_iter,
            steady_state_ridge=self.steady_state_ridge,
            allow_time_varying_fallback=self.allow_time_varying_fallback,
            steady_state_solution=self.steady_state_solution,
        )
        return macro.labels


@dataclass
class KalmanISSReconstructor(Reconstructor[GaussianPredictiveStateModel]):
    em_iters: int = 50
    em_tol: float = 1e-4
    em_ridge: float = 1e-6
    min_train_samples: int = 30
    macro_eps: float = 0.25
    macro_bins: int = 3
    macro_symboliser: str = "quantile"
    macro_builder: str = "hierarchical_complete"  # greedy | hierarchical_single | hierarchical_complete | linear_quantile
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
    model_select: bool = False
    selection_score: str = "predictive"  # predictive | stability
    selection_train_frac: float = 0.75
    selection_projection_modes: tuple[str, ...] = ("pca", "random", "psi_opt")
    selection_macro_builders: tuple[str, ...] = (
        "hierarchical_single",
        "hierarchical_complete",
        "linear_quantile",
        "greedy",
    )
    selection_macro_eps: tuple[float, ...] = ()
    selection_repeats: int = 4
    selection_seed_stride: int = 97
    selection_perturb: str = "seed_blocked"  # seed | blocked | seed_blocked
    selection_block_frac: float = 0.85

    @property
    def name(self) -> str:
        return "kalman_iss"

    @property
    def eps(self) -> float:
        # Kept for backwards-compatible CSV schema.
        return float(self.macro_eps)

    def _projection(self, y_train: Array, macro_dim: int, seed: int, *, mode: str) -> tuple[Array, str]:
        if mode == "pca":
            return _projection_pca(y_train, macro_dim), "pca"
        if mode == "random":
            return _projection_random(y_train.shape[1], macro_dim, seed=seed), "random"
        if mode == "psi_opt":
            # placeholder projection; caller replaces with optimised L.
            return _projection_pca(y_train, macro_dim), "psi_opt"
        raise ValueError(f"Unknown projection_mode={mode!r}. Expected 'pca', 'random', or 'psi_opt'.")

    def _fit_iss(
        self,
        y_train: Array,
        *,
        latent_dim: int,
        seed: int,
    ) -> tuple[KalmanISSModel, bool, Optional[SteadyStateKalmanSolution], float]:
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
        avg_train_nll = _average_gaussian_nll(y_train, mu_train, s_train)
        return iss_model, steady_state_enabled, steady_state_solution, avg_train_nll

    def _projection_with_psi(
        self,
        y_train: Array,
        iss_model: KalmanISSModel,
        *,
        macro_dim: int,
        seed: int,
        mode: str,
    ) -> tuple[Array, str, float, str, int, int, Array]:
        projection, projection_label = self._projection(y_train, macro_dim=macro_dim, seed=seed, mode=mode)
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

        should_optimise_psi = self.compute_psi or mode == "psi_opt"
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
        return projection, projection_label, float(psi_opt), psi_optimiser, psi_restarts, psi_iterations, psi_L

    def _candidate_specs(self) -> list[CandidateSpec]:
        if not self.model_select:
            return [
                CandidateSpec(
                    self.projection_mode,
                    _normalise_macro_builder(self.macro_builder),
                    self.macro_eps,
                )
            ]
        eps_values = self.selection_macro_eps if self.selection_macro_eps else (self.macro_eps,)
        specs: list[CandidateSpec] = []
        seen: set[tuple[str, str, float]] = set()
        for mode in self.selection_projection_modes:
            for builder in self.selection_macro_builders:
                norm_builder = _normalise_macro_builder(builder)
                for eps in eps_values:
                    key = (mode, norm_builder, float(eps))
                    if key in seen:
                        continue
                    seen.add(key)
                    specs.append(CandidateSpec(mode, norm_builder, float(eps)))
        if not specs:
            raise ValueError("Model selection requested but no candidate configurations were provided.")
        return specs

    def _normalise_scores(self, score: CandidateScore) -> tuple[float, float]:
        pred = score.predictive if math.isfinite(score.predictive) else math.inf
        stab = score.stability if math.isfinite(score.stability) else -math.inf
        return pred, stab

    def _select_candidate(
        self,
        y_train: Array,
        *,
        latent_dim: int,
        macro_dim: int,
        seed: int,
    ) -> CandidateScore:
        y_fit, y_val = _blocked_inner_split(
            y_train,
            frac=self.selection_train_frac,
            min_fit=max(self.min_train_samples, 5 * latent_dim),
        )
        specs = self._candidate_specs()
        repeats = max(1, self.selection_repeats)
        if self.selection_score == "stability":
            repeats = max(2, repeats)
        use_seed = self.selection_perturb in {"seed", "seed_blocked"}
        use_blocked = self.selection_perturb in {"blocked", "seed_blocked"}
        fit_floor = max(3, min(y_fit.shape[0], max(self.min_train_samples, 5 * latent_dim)))

        scored: list[CandidateScore] = []
        for spec in specs:
            labels: list[Array] = []
            pred_vals: list[float] = []
            n_states_vals: list[float] = []
            unif_vals: list[float] = []
            branch_vals: list[float] = []
            macro_times: list[float] = []
            mem_vals: list[float] = []
            n_pred_vals: list[float] = []

            for r in range(repeats):
                run_seed = seed + r * self.selection_seed_stride if use_seed else seed
                y_fit_run = y_fit
                if use_blocked:
                    y_fit_run = _blocked_resample(
                        y_fit,
                        frac=self.selection_block_frac,
                        min_len=fit_floor,
                        seed=seed + r * self.selection_seed_stride + 123_978,
                    )
                iss_model, steady_state_enabled, steady_state_solution, _ = self._fit_iss(
                    y_fit_run,
                    latent_dim=latent_dim,
                    seed=run_seed,
                )
                projection, _, _, _, _, _, _ = self._projection_with_psi(
                    y_fit_run,
                    iss_model,
                    macro_dim=macro_dim,
                    seed=run_seed,
                    mode=spec.projection_mode,
                )
                macro = _build_macro_dynamics(
                    y_train=y_fit_run,
                    iss_model=iss_model,
                    projection=projection,
                    eps=spec.macro_eps,
                    macro_bins=self.macro_bins,
                    macro_symboliser=self.macro_symboliser,
                    macro_builder=spec.macro_builder,
                    steady_state=steady_state_enabled,
                    steady_state_tol=self.steady_state_tol,
                    steady_state_max_iter=self.steady_state_max_iter,
                    steady_state_ridge=self.steady_state_ridge,
                    allow_time_varying_fallback=self.allow_time_varying_fallback,
                    steady_state_solution=steady_state_solution,
                )
                mu_fit, s_fit, _ = one_step_predictive_y(
                    y_fit,
                    iss_model,
                    steady_state=steady_state_enabled,
                    steady_state_tol=self.steady_state_tol,
                    steady_state_max_iter=self.steady_state_max_iter,
                    steady_state_ridge=self.steady_state_ridge,
                    steady_state_strict=not self.allow_time_varying_fallback,
                    allow_time_varying_fallback=self.allow_time_varying_fallback,
                    steady_state_solution=steady_state_solution,
                )
                ref_mu, ref_cov = _project_predictors(mu_fit, s_fit, projection, 0, y_fit.shape[0] - 2)
                labels.append(_assign_states(ref_mu, ref_cov, macro.centres_mu, macro.centres_cov))
                pred_nll = _macro_validation_nll(
                    y_fit,
                    y_val,
                    iss_model,
                    projection,
                    macro,
                    steady_state=steady_state_enabled,
                    steady_state_tol=self.steady_state_tol,
                    steady_state_max_iter=self.steady_state_max_iter,
                    steady_state_ridge=self.steady_state_ridge,
                    allow_time_varying_fallback=self.allow_time_varying_fallback,
                    steady_state_solution=steady_state_solution,
                )
                if math.isfinite(pred_nll):
                    pred_vals.append(pred_nll)
                n_states_vals.append(float(macro.n_macro_states))
                unif_vals.append(float(macro.unifilarity))
                branch_vals.append(float(macro.branch_entropy))
                macro_times.append(float(macro.build_time_s))
                mem_vals.append(float(macro.distance_matrix_mb_est))
                n_pred_vals.append(float(macro.n_pred))

            predictive = _mean(pred_vals, default=math.inf)
            if not math.isfinite(predictive):
                predictive = math.inf
            stability = _mean_pairwise_ari(labels)
            if not math.isfinite(stability):
                stability = -math.inf
            chosen = predictive if self.selection_score == "predictive" else stability
            scored.append(
                CandidateScore(
                    spec=spec,
                    predictive=predictive,
                    stability=stability,
                    chosen=chosen,
                    n_states_var=_variance(n_states_vals),
                    unifilarity_var=_variance(unif_vals),
                    branch_entropy_var=_variance(branch_vals),
                    n_states_mean=_mean(n_states_vals),
                    unifilarity_mean=_mean(unif_vals),
                    branch_entropy_mean=_mean(branch_vals),
                    macro_time_mean_s=_mean(macro_times),
                    macro_time_max_s=_max(macro_times),
                    distance_matrix_mb_est=_max(mem_vals, default=0.0),
                    n_pred_mean=_mean(n_pred_vals),
                )
            )

        if self.selection_score == "predictive":
            return min(
                scored,
                key=lambda row: (
                    self._normalise_scores(row)[0],
                    -self._normalise_scores(row)[1],
                    row.n_states_var + row.unifilarity_var + row.branch_entropy_var,
                    row.macro_time_mean_s if math.isfinite(row.macro_time_mean_s) else math.inf,
                    row.distance_matrix_mb_est if math.isfinite(row.distance_matrix_mb_est) else math.inf,
                ),
            )
        return min(
            scored,
            key=lambda row: (
                -self._normalise_scores(row)[1],
                row.n_states_var + row.unifilarity_var + row.branch_entropy_var,
                self._normalise_scores(row)[0],
                row.macro_time_mean_s if math.isfinite(row.macro_time_mean_s) else math.inf,
                row.distance_matrix_mb_est if math.isfinite(row.distance_matrix_mb_est) else math.inf,
            ),
        )

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
        if not _valid_macro_builder(self.macro_builder):
            raise ValueError(
                "macro_builder must be one of 'greedy', 'hierarchical_single', "
                "'hierarchical_complete', or 'linear_quantile' (global aliases are accepted)."
            )
        if self.iss_mode not in {"steady_state", "time_varying"}:
            raise ValueError("iss_mode must be 'steady_state' or 'time_varying'.")
        if self.steady_state_tol <= 0.0:
            raise ValueError("steady_state_tol must be > 0.")
        if self.steady_state_max_iter < 1:
            raise ValueError("steady_state_max_iter must be >= 1.")
        if self.steady_state_ridge < 0.0:
            raise ValueError("steady_state_ridge must be >= 0.")
        if self.projection_mode not in {"pca", "random", "psi_opt"}:
            raise ValueError("projection_mode must be 'pca', 'random', or 'psi_opt'.")
        if self.selection_score not in {"predictive", "stability"}:
            raise ValueError("selection_score must be 'predictive' or 'stability'.")
        if self.selection_repeats < 1:
            raise ValueError("selection_repeats must be >= 1.")
        if self.selection_seed_stride < 1:
            raise ValueError("selection_seed_stride must be >= 1.")
        if self.selection_perturb not in {"seed", "blocked", "seed_blocked"}:
            raise ValueError("selection_perturb must be 'seed', 'blocked', or 'seed_blocked'.")
        if not (0.55 <= self.selection_block_frac <= 1.0):
            raise ValueError("selection_block_frac must lie in [0.55, 1.0].")
        for mode in self.selection_projection_modes:
            if mode not in {"pca", "random", "psi_opt"}:
                raise ValueError(f"Unknown mode in selection_projection_modes: {mode!r}.")
        for builder in self.selection_macro_builders:
            if not _valid_macro_builder(builder):
                raise ValueError(f"Unknown builder in selection_macro_builders: {builder!r}.")
        for eps in self.selection_macro_eps:
            if eps < 0.0:
                raise ValueError("selection_macro_eps entries must be >= 0.")

        if self.model_select:
            picked = self._select_candidate(
                y_train,
                latent_dim=latent_dim,
                macro_dim=macro_dim,
                seed=seed,
            )
            chosen = picked.spec
            picked_predictive = picked.predictive
            picked_stability = picked.stability
            picked_value = picked.chosen
            picked_n_states_var = picked.n_states_var
            picked_unif_var = picked.unifilarity_var
            picked_branch_var = picked.branch_entropy_var
            picked_n_states_mean = picked.n_states_mean
            picked_unif_mean = picked.unifilarity_mean
            picked_branch_mean = picked.branch_entropy_mean
            picked_macro_time_mean = picked.macro_time_mean_s
            picked_macro_time_max = picked.macro_time_max_s
            picked_dist_mb = picked.distance_matrix_mb_est
            picked_n_pred = picked.n_pred_mean
        else:
            chosen = CandidateSpec(
                projection_mode=self.projection_mode,
                macro_builder=_normalise_macro_builder(self.macro_builder),
                macro_eps=self.macro_eps,
            )
            picked_predictive = math.nan
            picked_stability = math.nan
            picked_value = math.nan
            picked_n_states_var = math.nan
            picked_unif_var = math.nan
            picked_branch_var = math.nan
            picked_n_states_mean = math.nan
            picked_unif_mean = math.nan
            picked_branch_mean = math.nan
            picked_macro_time_mean = math.nan
            picked_macro_time_max = math.nan
            picked_dist_mb = math.nan
            picked_n_pred = math.nan

        iss_model, steady_state_enabled, steady_state_solution, avg_train_nll = self._fit_iss(
            y_train,
            latent_dim=latent_dim,
            seed=seed,
        )
        projection, projection_label, psi_opt, psi_optimiser, psi_restarts, psi_iterations, psi_L = self._projection_with_psi(
            y_train,
            iss_model,
            macro_dim=macro_dim,
            seed=seed,
            mode=chosen.projection_mode,
        )
        macro = _build_macro_dynamics(
            y_train=y_train,
            iss_model=iss_model,
            projection=projection,
            eps=chosen.macro_eps,
            macro_bins=self.macro_bins,
            macro_symboliser=self.macro_symboliser,
            macro_builder=chosen.macro_builder,
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
            n_macro_states=macro.n_macro_states,
            train_avg_nll=avg_train_nll,
            pi=macro.pi,
            transitions=macro.transitions,
            sa_counts=macro.sa_counts,
            projection=projection,
            projection_mode=projection_label,
            macro_builder=chosen.macro_builder,
            macro_eps=chosen.macro_eps,
            macro_bins=self.macro_bins,
            macro_symboliser=self.macro_symboliser,
            model_selection=self.model_select,
            selection_score=self.selection_score,
            selection_value=float(picked_value),
            selection_predictive=float(picked_predictive),
            selection_stability=float(picked_stability),
            selection_n_states_var=float(picked_n_states_var),
            selection_unifilarity_var=float(picked_unif_var),
            selection_branch_entropy_var=float(picked_branch_var),
            selection_n_states_mean=float(picked_n_states_mean),
            selection_unifilarity_mean=float(picked_unif_mean),
            selection_branch_entropy_mean=float(picked_branch_mean),
            selection_macro_time_mean_s=float(picked_macro_time_mean),
            selection_macro_time_max_s=float(picked_macro_time_max),
            selection_distance_matrix_mb_est=float(picked_dist_mb),
            selection_n_pred=float(picked_n_pred),
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
            macro_build_time_s=float(macro.build_time_s),
            macro_distance_matrix_mb_est=float(macro.distance_matrix_mb_est),
            macro_n_pred=int(macro.n_pred),
        )
