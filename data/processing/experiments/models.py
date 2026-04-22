"""Trial-wise EEG baselines built around linear one-step prediction."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Callable, Iterable

import numpy as np
from scipy.linalg import svd


DEFAULT_REP_DIMS = (2, 4, 8, 16, 32)
DEFAULT_CONTROL_KINDS = ("random_projection", "shuffled_dynamics", "observed_var1")
DEFAULT_PRISM_PROJECTION_MODES = ("pca", "psi_opt")
DEFAULT_PRISM_INPUT_SPACE = "pca_latent"
EPS = 1e-8


def split_trial_time(
    X: np.ndarray,
    train_fraction: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Expected trial with shape (time, channels), got {X.shape}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if not np.isfinite(X).all():
        raise ValueError("Trial contains NaN or infinite values")

    n_time = X.shape[0]
    split_idx = int(np.floor(n_time * train_fraction))
    split_idx = max(2, split_idx)
    split_idx = min(split_idx, n_time - 1)

    if split_idx < 2 or n_time - split_idx < 1:
        raise ValueError(
            "Trial is too short for a train/test split with VAR(1) evaluation"
        )

    return X[:split_idx], X[split_idx:]


def standardise_from_train(
    train_X: np.ndarray,
    test_X: np.ndarray,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_X.mean(axis=0, keepdims=True)
    scale = train_X.std(axis=0, keepdims=True)
    scale = np.where(scale < eps, 1.0, scale)
    return (train_X - mean) / scale, (test_X - mean) / scale, mean, scale


def fit_pca_projection(
    train_X: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_components < 1:
        raise ValueError("n_components must be positive")

    pca_mean = train_X.mean(axis=0, keepdims=True)
    centred = train_X - pca_mean
    max_rank = min(centred.shape[0], centred.shape[1])

    if n_components > max_rank:
        raise ValueError(
            f"rep_dim={n_components} exceeds PCA rank limit {max_rank} "
            f"for train shape {train_X.shape}"
        )

    _, _, vt = svd(centred, full_matrices=False, check_finite=False)
    components = vt[:n_components]
    return pca_mean, components


def random_orthonormal_components(
    n_features: int,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    if n_components < 1:
        raise ValueError("n_components must be positive")
    if n_components > n_features:
        raise ValueError(
            f"rep_dim={n_components} exceeds the feature count {n_features}"
        )

    rng = np.random.default_rng(random_state)
    basis = rng.standard_normal((n_features, n_components))
    q, _ = np.linalg.qr(basis)
    return q.T


def project_with_components(
    X: np.ndarray,
    projection_mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    return (X - projection_mean) @ components.T


def reconstruct_from_components(
    Z: np.ndarray,
    projection_mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    return Z @ components + projection_mean


def project_with_pca(
    X: np.ndarray,
    pca_mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    return project_with_components(X, pca_mean, components)


def reconstruct_from_pca(
    Z: np.ndarray,
    pca_mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    return reconstruct_from_components(Z, pca_mean, components)


def unstandardise(
    X_standardised: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return X_standardised * scale + mean


def select_evenly_spaced_indices(n_items: int, n_select: int) -> np.ndarray:
    if n_select < 1:
        raise ValueError("n_select must be at least 1")
    if n_select >= n_items:
        return np.arange(n_items, dtype=int)
    return np.linspace(0, n_items - 1, num=n_select, dtype=int)


def fit_var1_from_pairs(
    prev_Z: np.ndarray,
    next_Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prev_Z = np.asarray(prev_Z, dtype=float)
    next_Z = np.asarray(next_Z, dtype=float)
    if prev_Z.ndim != 2 or next_Z.ndim != 2 or prev_Z.shape != next_Z.shape:
        raise ValueError("prev_Z and next_Z must be matching 2D arrays")
    if prev_Z.shape[0] < 1:
        raise ValueError("At least one training transition is required")

    design = np.column_stack([np.ones(prev_Z.shape[0]), prev_Z])
    coef, _, _, _ = np.linalg.lstsq(design, next_Z, rcond=None)

    fitted = design @ coef
    residuals = next_Z - fitted
    cov = np.cov(residuals, rowvar=False, bias=False)
    cov = np.atleast_2d(cov).astype(float, copy=False)
    cov = 0.5 * (cov + cov.T)
    cov.flat[:: cov.shape[0] + 1] += 1e-6
    return coef, cov


def fit_var1(
    train_Z: np.ndarray,
    max_train_pairs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if train_Z.ndim != 2 or train_Z.shape[0] < 2:
        raise ValueError("train_Z must have at least 2 time points")

    prev_Z = train_Z[:-1]
    next_Z = train_Z[1:]
    if max_train_pairs is not None:
        if max_train_pairs < 1:
            raise ValueError("max_train_pairs must be at least 1")
        pair_idx = select_evenly_spaced_indices(prev_Z.shape[0], max_train_pairs)
        prev_Z = prev_Z[pair_idx]
        next_Z = next_Z[pair_idx]

    return fit_var1_from_pairs(prev_Z, next_Z)


def predict_var1(prev_Z: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(prev_Z.shape[0]), prev_Z])
    return design @ coef


def shuffle_time_axis(X: np.ndarray, random_state: int) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D data, got {X.shape}")

    rng = np.random.default_rng(random_state)
    order = rng.permutation(X.shape[0])
    return X[order]


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = y_true - y_pred
    centred = y_true - y_true.mean(axis=0, keepdims=True)
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum(centred * centred))

    if ss_tot <= 0.0:
        return float("nan")

    return 1.0 - (ss_res / ss_tot)


def mean_gaussian_nll(residuals: np.ndarray, cov: np.ndarray) -> float:
    dim = residuals.shape[1]
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    cov.flat[:: cov.shape[0] + 1] += 1e-6

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        diag = np.clip(np.diag(cov), 1e-6, None)
        inv_cov = np.diag(1.0 / diag)
        logdet = float(np.sum(np.log(diag)))
    else:
        inv_cov = np.linalg.pinv(cov)

    quad = np.einsum("ij,jk,ik->i", residuals, inv_cov, residuals)
    const = dim * np.log(2.0 * np.pi)
    return float(0.5 * np.mean(const + logdet + quad))


def gaussian_nll_per_step(
    y_true: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)

    if y_true.shape != mean.shape:
        raise ValueError(
            f"Gaussian NLL expects matching vectors, got {y_true.shape} and {mean.shape}"
        )

    cov = 0.5 * (cov + cov.T)
    cov.flat[:: cov.shape[0] + 1] += 1e-6

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0.0:
        diag = np.clip(np.diag(cov), 1e-6, None)
        inv_cov = np.diag(1.0 / diag)
        logdet = float(np.sum(np.log(diag)))
    else:
        inv_cov = np.linalg.pinv(cov)

    residual = y_true - mean
    quad = float(residual @ inv_cov @ residual)
    const = y_true.size * math.log(2.0 * math.pi)
    return float(0.5 * (const + logdet + quad))


def mean_sequence_gaussian_nll(
    y_true: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)

    if y_true.shape != means.shape:
        raise ValueError(
            f"Sequence NLL expects matching shapes, got {y_true.shape} and {means.shape}"
        )
    if covariances.shape != (y_true.shape[0], y_true.shape[1], y_true.shape[1]):
        raise ValueError(
            "Sequence NLL expects covariances with shape "
            f"({y_true.shape[0]}, {y_true.shape[1]}, {y_true.shape[1]}), "
            f"got {covariances.shape}"
        )

    losses = [
        gaussian_nll_per_step(y_true[t], means[t], covariances[t])
        for t in range(y_true.shape[0])
    ]
    return float(np.mean(losses))


def score_var1_model(
    train_model_space: np.ndarray,
    test_model_space: np.ndarray,
    reconstruct_to_observed: Callable[[np.ndarray], np.ndarray],
    test_observed: np.ndarray,
    initial_prev: np.ndarray | None = None,
    max_train_pairs: int | None = None,
) -> dict[str, float]:
    coef, cov = fit_var1(train_model_space, max_train_pairs=max_train_pairs)
    if initial_prev is None:
        initial_prev = train_model_space[-1:]

    prev_test = np.vstack([initial_prev, test_model_space[:-1]])
    pred_test = predict_var1(prev_test, coef)
    residuals = test_model_space - pred_test
    pred_observed = reconstruct_to_observed(pred_test)

    return {
        "pred_mse_obs": float(np.mean((test_observed - pred_observed) ** 2)),
        "pred_r2_obs": compute_r2(test_observed, pred_observed),
        "pred_mse_latent": float(np.mean(residuals * residuals)),
        "pred_r2_latent": compute_r2(test_model_space, pred_test),
        "pred_nll_latent": mean_gaussian_nll(residuals, cov),
    }


def score_gaussian_observed_predictions(
    test_observed: np.ndarray,
    pred_observed: np.ndarray,
    pred_covariances: np.ndarray,
    *,
    include_nll: bool = True,
) -> dict[str, float]:
    metrics = {
        "pred_mse_obs": float(np.mean((test_observed - pred_observed) ** 2)),
        "pred_r2_obs": compute_r2(test_observed, pred_observed),
        "pred_nll_obs": np.nan,
    }
    if include_nll:
        metrics["pred_nll_obs"] = mean_sequence_gaussian_nll(
            test_observed,
            pred_observed,
            pred_covariances,
        )
    return metrics


def _load_prism_components():
    src_root = Path(__file__).resolve().parents[3] / "src"
    if not src_root.exists():
        raise FileNotFoundError(f"PRISM source directory not found: {src_root}")
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from prism.reconstruction.kalman_iss import KalmanISSReconstructor
    from prism.representations.continuous import ISSDim

    return KalmanISSReconstructor, ISSDim


def unstandardise_gaussian_predictions(
    pred_mean_standardised: np.ndarray,
    pred_cov_standardised: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_mean_standardised = np.asarray(pred_mean_standardised, dtype=float)
    pred_cov_standardised = np.asarray(pred_cov_standardised, dtype=float)
    mean_vec = np.asarray(mean, dtype=float).reshape(-1)
    scale_vec = np.asarray(scale, dtype=float).reshape(-1)

    if pred_mean_standardised.ndim == 3 and pred_mean_standardised.shape[-1] == 1:
        pred_mean_standardised = pred_mean_standardised[..., 0]
    elif pred_mean_standardised.ndim != 2:
        raise ValueError(
            "Expected predicted means with shape (time, channels) or (time, channels, 1), "
            f"got {pred_mean_standardised.shape}"
        )

    pred_mean_observed = pred_mean_standardised * scale_vec[None, :] + mean_vec[None, :]
    scale_outer = scale_vec[:, None] * scale_vec[None, :]
    pred_cov_observed = pred_cov_standardised * scale_outer[None, :, :]
    return pred_mean_observed, pred_cov_observed


def reconstruct_gaussian_from_components(
    pred_mean_latent: np.ndarray,
    pred_cov_latent: np.ndarray,
    projection_mean: np.ndarray,
    components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_mean_latent = np.asarray(pred_mean_latent, dtype=float)
    pred_cov_latent = np.asarray(pred_cov_latent, dtype=float)
    projection_mean = np.asarray(projection_mean, dtype=float)
    components = np.asarray(components, dtype=float)

    if pred_mean_latent.ndim == 3 and pred_mean_latent.shape[-1] == 1:
        pred_mean_latent = pred_mean_latent[..., 0]
    elif pred_mean_latent.ndim != 2:
        raise ValueError(
            "Expected latent means with shape (time, components) or "
            f"(time, components, 1), got {pred_mean_latent.shape}"
        )

    pred_mean = reconstruct_from_components(pred_mean_latent, projection_mean, components)
    pred_cov = np.empty(
        (pred_cov_latent.shape[0], components.shape[1], components.shape[1]),
        dtype=float,
    )
    for idx in range(pred_cov_latent.shape[0]):
        pred_cov[idx] = components.T @ pred_cov_latent[idx] @ components
    return pred_mean, pred_cov


def make_error_rows(
    X: np.ndarray,
    rep_dims: Iterable[int],
    train_len: int,
    test_len: int,
    error: str,
    extra_fields: dict | None = None,
) -> list[dict]:
    n_time, n_channels = X.shape
    extra_fields = {} if extra_fields is None else dict(extra_fields)
    rows = []
    for rep_dim in rep_dims:
        rows.append(
            {
                **extra_fields,
                "rep_dim": int(rep_dim),
                "n_time": int(n_time),
                "n_channels": int(n_channels),
                "train_len": int(train_len),
                "test_len": int(test_len),
                "pred_mse_obs": np.nan,
                "pred_r2_obs": np.nan,
                "pred_mse_latent": np.nan,
                "pred_r2_latent": np.nan,
                "pred_nll_latent": np.nan,
                "error": error,
            }
        )
    return rows


def evaluate_baseline_train_test(
    train_X: np.ndarray,
    test_X: np.ndarray,
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    n_time: int | None = None,
    max_train_pairs: int | None = None,
) -> list[dict]:
    train_X = np.asarray(train_X, dtype=float)
    test_X = np.asarray(test_X, dtype=float)
    rep_dims = tuple(int(dim) for dim in rep_dims)

    if train_X.ndim != 2 or test_X.ndim != 2:
        raise ValueError("train_X and test_X must both have shape (time, channels)")
    if train_X.shape[1] != test_X.shape[1]:
        raise ValueError(
            "train_X and test_X must have the same number of channels, got "
            f"{train_X.shape[1]} and {test_X.shape[1]}"
        )
    if train_X.shape[0] < 2:
        raise ValueError("train_X must contain at least 2 time points")
    if test_X.shape[0] < 1:
        raise ValueError("test_X must contain at least 1 time point")
    if not np.isfinite(train_X).all() or not np.isfinite(test_X).all():
        raise ValueError("train_X and test_X must not contain NaN or infinite values")

    total_n_time = int(train_X.shape[0] + test_X.shape[0] if n_time is None else n_time)
    n_channels = int(train_X.shape[1])
    train_len = int(train_X.shape[0])
    test_len = int(test_X.shape[0])
    train_pair_count = (
        train_len - 1
        if max_train_pairs is None
        else int(min(train_len - 1, max_train_pairs))
    )

    train_Zsc, test_Zsc, train_mean, train_scale = standardise_from_train(train_X, test_X)
    rows = []

    for rep_dim in rep_dims:
        row = {
            "rep_dim": int(rep_dim),
            "n_time": total_n_time,
            "n_channels": n_channels,
            "train_len": train_len,
            "test_len": test_len,
            "train_pair_count": train_pair_count,
            "pred_mse_obs": np.nan,
            "pred_r2_obs": np.nan,
            "pred_mse_latent": np.nan,
            "pred_r2_latent": np.nan,
            "pred_nll_latent": np.nan,
            "error": "",
        }

        try:
            pca_mean, components = fit_pca_projection(train_Zsc, n_components=rep_dim)
            train_latent = project_with_pca(train_Zsc, pca_mean, components)
            test_latent = project_with_pca(test_Zsc, pca_mean, components)
            row.update(
                score_var1_model(
                    train_model_space=train_latent,
                    test_model_space=test_latent,
                    reconstruct_to_observed=lambda Z: unstandardise(
                        reconstruct_from_pca(Z, pca_mean, components),
                        train_mean,
                        train_scale,
                    ),
                    test_observed=test_X,
                    max_train_pairs=max_train_pairs,
                )
            )
        except Exception as exc:
            row["error"] = str(exc)

        rows.append(row)

    return rows


def make_prism_error_rows(
    X: np.ndarray,
    rep_dims: Iterable[int],
    projection_modes: Iterable[str],
    train_len: int,
    test_len: int,
    error: str,
) -> list[dict]:
    n_time, n_channels = X.shape
    rows = []
    for projection_mode in projection_modes:
        for rep_dim in rep_dims:
            rows.append(
                {
                    "model_family": f"prism_{projection_mode}",
                    "projection_mode": projection_mode,
                    "input_space": DEFAULT_PRISM_INPUT_SPACE,
                    "rep_dim": int(rep_dim),
                    "latent_dim": int(rep_dim),
                    "macro_dim": int(rep_dim),
                    "n_time": int(n_time),
                    "n_channels": int(n_channels),
                    "train_len": int(train_len),
                    "test_len": int(test_len),
                    "pred_mse_obs": np.nan,
                    "pred_r2_obs": np.nan,
                    "pred_nll_obs": np.nan,
                    "macro_logloss": np.nan,
                    "n_macro_states": np.nan,
                    "psi_opt": np.nan,
                    "macro_builder": "",
                    "iss_mode": "",
                    "error": error,
                }
            )
    return rows


def make_prism_train_test_error_rows(
    *,
    n_time: int,
    n_channels: int,
    rep_dims: Iterable[int],
    projection_modes: Iterable[str],
    train_len: int,
    test_len: int,
    error: str,
) -> list[dict]:
    rows = []
    for projection_mode in projection_modes:
        for rep_dim in rep_dims:
            rows.append(
                {
                    "model_family": f"prism_{projection_mode}",
                    "projection_mode": projection_mode,
                    "input_space": DEFAULT_PRISM_INPUT_SPACE,
                    "rep_dim": int(rep_dim),
                    "latent_dim": int(rep_dim),
                    "macro_dim": int(rep_dim),
                    "n_time": int(n_time),
                    "n_channels": int(n_channels),
                    "train_len": int(train_len),
                    "test_len": int(test_len),
                    "pred_mse_obs": np.nan,
                    "pred_r2_obs": np.nan,
                    "pred_nll_obs": np.nan,
                    "macro_logloss": np.nan,
                    "n_macro_states": np.nan,
                    "psi_opt": np.nan,
                    "macro_builder": "",
                    "iss_mode": "",
                    "error": error,
                }
            )
    return rows


def evaluate_trial_baseline(
    X: np.ndarray,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    train_fraction: float = 0.7,
) -> list[dict]:
    """Run the PCA + VAR(1) baseline for one trial."""
    rep_dims = tuple(int(dim) for dim in rep_dims)
    X = np.asarray(X, dtype=float)

    try:
        train_X, test_X = split_trial_time(X, train_fraction=train_fraction)
    except Exception as exc:
        train_len = max(0, min(int(np.floor(X.shape[0] * train_fraction)), X.shape[0]))
        test_len = max(0, X.shape[0] - train_len)
        return make_error_rows(X, rep_dims, train_len, test_len, str(exc))
    return evaluate_baseline_train_test(
        train_X,
        test_X,
        rep_dims=rep_dims,
        n_time=int(X.shape[0]),
    )


def _load_iss_scoring_components():
    src_root = Path(__file__).resolve().parents[3] / "src"
    if not src_root.exists():
        raise FileNotFoundError(f"PRISM source directory not found: {src_root}")
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from prism.continuous.iss import (
        KalmanISSConfig,
        fit_kalman_iss_em,
        one_step_predictive_y,
    )
    return fit_kalman_iss_em, KalmanISSConfig, one_step_predictive_y


def evaluate_iss_train_test(
    train_X: np.ndarray,
    test_X: np.ndarray,
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    n_time: int | None = None,
    em_iters: int = 50,
    em_tol: float = 1e-4,
    em_ridge: float = 1e-6,
    allow_time_varying_fallback: bool = False,
) -> list[dict]:
    train_X = np.asarray(train_X, dtype=float)
    test_X = np.asarray(test_X, dtype=float)
    rep_dims = tuple(int(dim) for dim in rep_dims)

    if train_X.ndim != 2 or test_X.ndim != 2:
        raise ValueError("train_X and test_X must both have shape (time, channels)")
    if train_X.shape[1] != test_X.shape[1]:
        raise ValueError(
            "train_X and test_X must have the same number of channels, "
            f"got {train_X.shape[1]} and {test_X.shape[1]}"
        )
    if train_X.shape[0] < 5:
        raise ValueError("train_X must contain at least 5 time points for ISS EM fitting")
    if test_X.shape[0] < 1:
        raise ValueError("test_X must contain at least 1 time point")
    if not np.isfinite(train_X).all() or not np.isfinite(test_X).all():
        raise ValueError("train_X and test_X must not contain NaN or infinite values")

    fit_kalman_iss_em, KalmanISSConfig, one_step_predictive_y = _load_iss_scoring_components()

    total_n_time = int(train_X.shape[0] + test_X.shape[0] if n_time is None else n_time)
    n_channels = int(train_X.shape[1])
    train_len = int(train_X.shape[0])
    test_len = int(test_X.shape[0])

    train_Zsc, test_Zsc, train_mean, train_scale = standardise_from_train(train_X, test_X)
    rows = []

    for rep_dim in rep_dims:
        row = {
            "rep_dim": int(rep_dim),
            "n_time": total_n_time,
            "n_channels": n_channels,
            "train_len": train_len,
            "test_len": test_len,
            "train_pair_count": train_len,
            "pred_mse_obs": np.nan,
            "pred_r2_obs": np.nan,
            "pred_mse_latent": np.nan,
            "pred_r2_latent": np.nan,
            "pred_nll_latent": np.nan,
            "error": "",
        }

        try:
            pca_mean, components = fit_pca_projection(train_Zsc, n_components=rep_dim)
            train_latent = project_with_pca(train_Zsc, pca_mean, components)
            test_latent = project_with_pca(test_Zsc, pca_mean, components)

            cfg = KalmanISSConfig(
                latent_dim=rep_dim,
                em_iters=em_iters,
                tol=em_tol,
                ridge=em_ridge,
            )
            model = fit_kalman_iss_em(train_latent, cfg)

            # Predict over the full sequence so the Kalman filter warms up on
            # training data before producing test-segment predictions.
            full_latent = np.vstack([train_latent, test_latent])
            mu_y, S_y, _ = one_step_predictive_y(
                full_latent,
                model,
                steady_state=True,
                allow_time_varying_fallback=allow_time_varying_fallback,
            )

            # mu_y: (T_total, rep_dim, 1); S_y: (T_total, rep_dim, rep_dim)
            mu_y_test = mu_y[train_len:, :, 0]  # (test_len, rep_dim)
            S_y_test = S_y[train_len:]           # (test_len, rep_dim, rep_dim)

            pred_obs_zsc = reconstruct_from_pca(mu_y_test, pca_mean, components)
            pred_obs = unstandardise(pred_obs_zsc, train_mean, train_scale)

            row["pred_mse_obs"] = float(np.mean((test_X - pred_obs) ** 2))
            row["pred_r2_obs"] = compute_r2(test_X, pred_obs)
            row["pred_mse_latent"] = float(np.mean((test_latent - mu_y_test) ** 2))
            row["pred_r2_latent"] = compute_r2(test_latent, mu_y_test)
            row["pred_nll_latent"] = mean_sequence_gaussian_nll(
                test_latent, mu_y_test, S_y_test
            )

        except Exception as exc:
            row["error"] = str(exc)

        rows.append(row)

    return rows


def evaluate_trial_iss(
    X: np.ndarray,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    train_fraction: float = 0.7,
    em_iters: int = 50,
    em_tol: float = 1e-4,
    em_ridge: float = 1e-6,
    allow_time_varying_fallback: bool = False,
) -> list[dict]:
    """Run the PCA + steady-state ISS evaluator for one trial."""
    rep_dims = tuple(int(dim) for dim in rep_dims)
    X = np.asarray(X, dtype=float)

    try:
        train_X, test_X = split_trial_time(X, train_fraction=train_fraction)
    except Exception as exc:
        train_len = max(0, min(int(np.floor(X.shape[0] * train_fraction)), X.shape[0]))
        test_len = max(0, X.shape[0] - train_len)
        return make_error_rows(X, rep_dims, train_len, test_len, str(exc))
    return evaluate_iss_train_test(
        train_X,
        test_X,
        rep_dims=rep_dims,
        n_time=int(X.shape[0]),
        em_iters=em_iters,
        em_tol=em_tol,
        em_ridge=em_ridge,
        allow_time_varying_fallback=allow_time_varying_fallback,
    )


def evaluate_trial_controls(
    X: np.ndarray,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    control_kinds: Iterable[str] = DEFAULT_CONTROL_KINDS,
    train_fraction: float = 0.7,
    random_state: int = 0,
) -> list[dict]:
    """Run matched control baselines for one trial."""
    rep_dims = tuple(int(dim) for dim in rep_dims)
    control_kinds = tuple(str(kind) for kind in control_kinds)
    X = np.asarray(X, dtype=float)

    try:
        train_X, test_X = split_trial_time(X, train_fraction=train_fraction)
    except Exception as exc:
        train_len = max(0, min(int(np.floor(X.shape[0] * train_fraction)), X.shape[0]))
        test_len = max(0, X.shape[0] - train_len)
        rows: list[dict] = []
        for control_kind in control_kinds:
            control_rep_dims = (X.shape[1],) if control_kind == "observed_var1" else rep_dims
            rows.extend(
                make_error_rows(
                    X,
                    control_rep_dims,
                    train_len,
                    test_len,
                    str(exc),
                    extra_fields={
                        "control_kind": control_kind,
                        "control_seed": np.nan,
                    },
                )
            )
        return rows

    train_Zsc, test_Zsc, train_mean, train_scale = standardise_from_train(
        train_X,
        test_X,
    )
    train_len = train_X.shape[0]
    test_len = test_X.shape[0]
    rows = []
    rng = np.random.default_rng(random_state)

    for control_kind in control_kinds:
        control_rep_dims = (X.shape[1],) if control_kind == "observed_var1" else rep_dims

        for rep_dim in control_rep_dims:
            control_seed = (
                np.nan
                if control_kind == "observed_var1"
                else int(rng.integers(0, np.iinfo(np.int32).max))
            )
            row = {
                "control_kind": control_kind,
                "control_seed": control_seed,
                "rep_dim": int(rep_dim),
                "n_time": int(X.shape[0]),
                "n_channels": int(X.shape[1]),
                "train_len": int(train_len),
                "test_len": int(test_len),
                "pred_mse_obs": np.nan,
                "pred_r2_obs": np.nan,
                "pred_mse_latent": np.nan,
                "pred_r2_latent": np.nan,
                "pred_nll_latent": np.nan,
                "error": "",
            }

            try:
                if control_kind == "random_projection":
                    projection_mean = train_Zsc.mean(axis=0, keepdims=True)
                    components = random_orthonormal_components(
                        n_features=train_Zsc.shape[1],
                        n_components=rep_dim,
                        random_state=control_seed,
                    )
                    train_model_space = project_with_components(
                        train_Zsc,
                        projection_mean,
                        components,
                    )
                    test_model_space = project_with_components(
                        test_Zsc,
                        projection_mean,
                        components,
                    )
                    row.update(
                        score_var1_model(
                            train_model_space=train_model_space,
                            test_model_space=test_model_space,
                            reconstruct_to_observed=lambda Z: unstandardise(
                                reconstruct_from_components(Z, projection_mean, components),
                                train_mean,
                                train_scale,
                            ),
                            test_observed=test_X,
                        )
                    )
                elif control_kind == "shuffled_dynamics":
                    projection_mean, components = fit_pca_projection(
                        train_Zsc,
                        n_components=rep_dim,
                    )
                    train_model_space = project_with_pca(
                        train_Zsc,
                        projection_mean,
                        components,
                    )
                    test_model_space = project_with_pca(
                        test_Zsc,
                        projection_mean,
                        components,
                    )
                    shuffled_train = shuffle_time_axis(
                        train_model_space,
                        random_state=control_seed,
                    )
                    row.update(
                        score_var1_model(
                            train_model_space=shuffled_train,
                            test_model_space=test_model_space,
                            reconstruct_to_observed=lambda Z: unstandardise(
                                reconstruct_from_pca(Z, projection_mean, components),
                                train_mean,
                                train_scale,
                            ),
                            test_observed=test_X,
                            initial_prev=train_model_space[-1:],
                        )
                    )
                elif control_kind == "observed_var1":
                    row.update(
                        score_var1_model(
                            train_model_space=train_Zsc,
                            test_model_space=test_Zsc,
                            reconstruct_to_observed=lambda Z: unstandardise(
                                Z,
                                train_mean,
                                train_scale,
                            ),
                            test_observed=test_X,
                        )
                    )
                else:
                    raise ValueError(f"Unknown control_kind: {control_kind}")
            except Exception as exc:
                row["error"] = str(exc)

            rows.append(row)

    return rows


def evaluate_trial_prism(
    X: np.ndarray,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    projection_modes: Iterable[str] = DEFAULT_PRISM_PROJECTION_MODES,
    train_fraction: float = 0.7,
    *,
    macro_builder: str = "hierarchical_complete",
    macro_eps: float = 0.25,
    macro_bins: int = 3,
    em_iters: int = 20,
    em_tol: float = 1e-3,
    em_ridge: float = 1e-6,
    iss_mode: str = "steady_state",
    allow_time_varying_fallback: bool = False,
    input_space: str = DEFAULT_PRISM_INPUT_SPACE,
    random_state: int = 0,
) -> list[dict]:
    """Run the matched PRISM continuous model on one EEG trial."""
    rep_dims = tuple(int(dim) for dim in rep_dims)
    projection_modes = tuple(str(mode) for mode in projection_modes)
    X = np.asarray(X, dtype=float)

    try:
        train_X, test_X = split_trial_time(X, train_fraction=train_fraction)
    except Exception as exc:
        train_len = max(0, min(int(np.floor(X.shape[0] * train_fraction)), X.shape[0]))
        test_len = max(0, X.shape[0] - train_len)
        return make_prism_error_rows(
            X,
            rep_dims,
            projection_modes,
            train_len,
            test_len,
            str(exc),
        )

    train_Zsc, test_Zsc, train_mean, train_scale = standardise_from_train(train_X, test_X)
    train_len = train_X.shape[0]
    test_len = test_X.shape[0]

    try:
        KalmanISSReconstructor, ISSDim = _load_prism_components()
    except Exception as exc:
        return make_prism_error_rows(
            X,
            rep_dims,
            projection_modes,
            train_len,
            test_len,
            str(exc),
        )

    rows = []
    rng = np.random.default_rng(random_state)

    for projection_mode in projection_modes:
        for rep_dim in rep_dims:
            row = {
                "model_family": f"prism_{projection_mode}",
                "projection_mode": projection_mode,
                "input_space": input_space,
                "rep_dim": int(rep_dim),
                "latent_dim": int(rep_dim),
                "macro_dim": int(rep_dim),
                "n_time": int(X.shape[0]),
                "n_channels": int(X.shape[1]),
                "train_len": int(train_len),
                "test_len": int(test_len),
                "pred_mse_obs": np.nan,
                "pred_r2_obs": np.nan,
                "pred_nll_obs": np.nan,
                "macro_logloss": np.nan,
                "n_macro_states": np.nan,
                "psi_opt": np.nan,
                "macro_builder": macro_builder,
                "iss_mode": iss_mode,
                "error": "",
            }

            try:
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                if input_space == "raw_channels":
                    train_model_obs = train_Zsc
                    test_model_obs = test_Zsc
                    reconstruct_predictions = lambda mean, cov: unstandardise_gaussian_predictions(
                        mean,
                        cov,
                        train_mean,
                        train_scale,
                    )
                elif input_space == "pca_latent":
                    pca_mean, components = fit_pca_projection(
                        train_Zsc,
                        n_components=rep_dim,
                    )
                    train_model_obs = project_with_pca(train_Zsc, pca_mean, components)
                    test_model_obs = project_with_pca(test_Zsc, pca_mean, components)
                    reconstruct_predictions = lambda mean, cov: unstandardise_gaussian_predictions(
                        *reconstruct_gaussian_from_components(
                            mean,
                            cov,
                            pca_mean,
                            components,
                        ),
                        train_mean,
                        train_scale,
                    )
                else:
                    raise ValueError(
                        f"Unknown input_space: {input_space}. Expected 'raw_channels' or 'pca_latent'."
                    )

                reconstructor = KalmanISSReconstructor(
                    em_iters=em_iters,
                    em_tol=em_tol,
                    em_ridge=em_ridge,
                    macro_eps=macro_eps,
                    macro_bins=macro_bins,
                    macro_builder=macro_builder,
                    projection_mode=projection_mode,
                    iss_mode=iss_mode,
                    allow_time_varying_fallback=allow_time_varying_fallback,
                )
                model = reconstructor.fit(
                    train_model_obs,
                    ISSDim(d=rep_dim, dv=rep_dim),
                    seed=seed,
                )
                pred_mean_model, pred_cov_model = model.predictive_distributions(
                    test_model_obs,
                    context=train_model_obs,
                )
                pred_mean_observed, pred_cov_observed = reconstruct_predictions(
                    pred_mean_model,
                    pred_cov_model,
                )

                row.update(
                    score_gaussian_observed_predictions(
                        test_observed=test_X,
                        pred_observed=pred_mean_observed,
                        pred_covariances=pred_cov_observed,
                        include_nll=False,
                    )
                )
                row["macro_logloss"] = float(
                    model.macro_average_negative_log_likelihood(
                        test_model_obs,
                        context=train_model_obs,
                    )
                )
                row["model_family"] = f"prism_{model.projection_mode}"
                row["projection_mode"] = str(model.projection_mode)
                row["n_macro_states"] = int(model.n_macro_states)
                row["psi_opt"] = float(model.psi_opt)
                row["macro_builder"] = str(model.macro_builder)
                row["iss_mode"] = str(model.iss_mode)
            except Exception as exc:
                row["error"] = str(exc)

            rows.append(row)

    return rows


def evaluate_prism_train_test(
    train_X: np.ndarray,
    test_X: np.ndarray,
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    projection_modes: Iterable[str] = DEFAULT_PRISM_PROJECTION_MODES,
    n_time: int | None = None,
    macro_builder: str = "hierarchical_complete",
    macro_eps: float = 0.25,
    macro_bins: int = 3,
    em_iters: int = 20,
    em_tol: float = 1e-3,
    em_ridge: float = 1e-6,
    iss_mode: str = "steady_state",
    allow_time_varying_fallback: bool = False,
    input_space: str = DEFAULT_PRISM_INPUT_SPACE,
    random_state: int = 0,
) -> list[dict]:
    train_X = np.asarray(train_X, dtype=float)
    test_X = np.asarray(test_X, dtype=float)
    rep_dims = tuple(int(dim) for dim in rep_dims)
    projection_modes = tuple(str(mode) for mode in projection_modes)

    if train_X.ndim != 2 or test_X.ndim != 2:
        raise ValueError("train_X and test_X must both have shape (time, channels)")
    if train_X.shape[1] != test_X.shape[1]:
        raise ValueError(
            "train_X and test_X must have the same number of channels, got "
            f"{train_X.shape[1]} and {test_X.shape[1]}"
        )
    if train_X.shape[0] < 2:
        raise ValueError("train_X must contain at least 2 time points")
    if test_X.shape[0] < 1:
        raise ValueError("test_X must contain at least 1 time point")
    if not np.isfinite(train_X).all() or not np.isfinite(test_X).all():
        raise ValueError("train_X and test_X must not contain NaN or infinite values")

    total_n_time = int(train_X.shape[0] + test_X.shape[0] if n_time is None else n_time)
    n_channels = int(train_X.shape[1])
    train_len = int(train_X.shape[0])
    test_len = int(test_X.shape[0])

    train_Zsc, test_Zsc, train_mean, train_scale = standardise_from_train(train_X, test_X)

    try:
        KalmanISSReconstructor, ISSDim = _load_prism_components()
    except Exception as exc:
        return make_prism_train_test_error_rows(
            n_time=total_n_time,
            n_channels=n_channels,
            rep_dims=rep_dims,
            projection_modes=projection_modes,
            train_len=train_len,
            test_len=test_len,
            error=str(exc),
        )

    rows = []
    rng = np.random.default_rng(random_state)

    for projection_mode in projection_modes:
        for rep_dim in rep_dims:
            row = {
                "model_family": f"prism_{projection_mode}",
                "projection_mode": projection_mode,
                "input_space": input_space,
                "rep_dim": int(rep_dim),
                "latent_dim": int(rep_dim),
                "macro_dim": int(rep_dim),
                "n_time": total_n_time,
                "n_channels": n_channels,
                "train_len": train_len,
                "test_len": test_len,
                "pred_mse_obs": np.nan,
                "pred_r2_obs": np.nan,
                "pred_nll_obs": np.nan,
                "macro_logloss": np.nan,
                "n_macro_states": np.nan,
                "psi_opt": np.nan,
                "macro_builder": macro_builder,
                "iss_mode": iss_mode,
                "error": "",
            }

            try:
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                if input_space == "raw_channels":
                    train_model_obs = train_Zsc
                    test_model_obs = test_Zsc
                    reconstruct_predictions = lambda mean, cov: unstandardise_gaussian_predictions(
                        mean,
                        cov,
                        train_mean,
                        train_scale,
                    )
                elif input_space == "pca_latent":
                    pca_mean, components = fit_pca_projection(
                        train_Zsc,
                        n_components=rep_dim,
                    )
                    train_model_obs = project_with_pca(train_Zsc, pca_mean, components)
                    test_model_obs = project_with_pca(test_Zsc, pca_mean, components)
                    reconstruct_predictions = lambda mean, cov: unstandardise_gaussian_predictions(
                        *reconstruct_gaussian_from_components(
                            mean,
                            cov,
                            pca_mean,
                            components,
                        ),
                        train_mean,
                        train_scale,
                    )
                else:
                    raise ValueError(
                        f"Unknown input_space: {input_space}. Expected 'raw_channels' or 'pca_latent'."
                    )

                reconstructor = KalmanISSReconstructor(
                    em_iters=em_iters,
                    em_tol=em_tol,
                    em_ridge=em_ridge,
                    macro_eps=macro_eps,
                    macro_bins=macro_bins,
                    macro_builder=macro_builder,
                    projection_mode=projection_mode,
                    iss_mode=iss_mode,
                    allow_time_varying_fallback=allow_time_varying_fallback,
                )
                model = reconstructor.fit(
                    train_model_obs,
                    ISSDim(d=rep_dim, dv=rep_dim),
                    seed=seed,
                )
                pred_mean_model, pred_cov_model = model.predictive_distributions(
                    test_model_obs,
                    context=train_model_obs,
                )
                pred_mean_observed, pred_cov_observed = reconstruct_predictions(
                    pred_mean_model,
                    pred_cov_model,
                )

                row.update(
                    score_gaussian_observed_predictions(
                        test_observed=test_X,
                        pred_observed=pred_mean_observed,
                        pred_covariances=pred_cov_observed,
                        include_nll=False,
                    )
                )
                row["macro_logloss"] = float(
                    model.macro_average_negative_log_likelihood(
                        test_model_obs,
                        context=train_model_obs,
                    )
                )
                row["model_family"] = f"prism_{model.projection_mode}"
                row["projection_mode"] = str(model.projection_mode)
                row["n_macro_states"] = int(model.n_macro_states)
                row["psi_opt"] = float(model.psi_opt)
                row["macro_builder"] = str(model.macro_builder)
                row["iss_mode"] = str(model.iss_mode)
            except Exception as exc:
                row["error"] = str(exc)

            rows.append(row)

    return rows


def fit_iss_label_chains(
    train_X: np.ndarray,
    *,
    rep_dim: int = 4,
    eps_values: tuple = (0.10, 0.15, 0.25, 0.40),
    macro_bins: int = 3,
    macro_builder: str = "hierarchical_single",
    projection_mode: str = "pca",
    em_iters: int = 20,
    em_tol: float = 1e-3,
    em_ridge: float = 1e-6,
    iss_mode: str = "steady_state",
    allow_time_varying_fallback: bool = False,
    random_state: int = 0,
):
    """Fit one ISS model and return per-timestep macro labels at each eps.

    Returns a dict mapping eps -> label array (length T_train-1) suitable for
    building a CE 2.0 coarsening chain.  Returns None if fitting fails.
    """
    train_X = np.asarray(train_X, dtype=float)
    if train_X.shape[0] < max(30, 5 * rep_dim):
        return None

    try:
        KalmanISSReconstructor, ISSDim = _load_prism_components()
    except Exception:
        return None

    train_Zsc, _, _, _ = standardise_from_train(train_X, train_X[:1])

    if projection_mode == "pca":
        try:
            pca_mean, components = fit_pca_projection(train_Zsc, n_components=rep_dim)
            model_obs = project_with_pca(train_Zsc, pca_mean, components)
        except Exception:
            return None
    else:
        model_obs = train_Zsc

    reconstructor = KalmanISSReconstructor(
        em_iters=em_iters,
        em_tol=em_tol,
        em_ridge=em_ridge,
        macro_eps=float(eps_values[0]),
        macro_bins=macro_bins,
        macro_builder=macro_builder,
        projection_mode=projection_mode,
        iss_mode=iss_mode,
        allow_time_varying_fallback=allow_time_varying_fallback,
    )
    try:
        model = reconstructor.fit(
            model_obs,
            ISSDim(d=rep_dim, dv=rep_dim),
            seed=int(random_state),
        )
    except Exception:
        return None

    chains = {}
    for eps in eps_values:
        try:
            chains[float(eps)] = model.macro_label_sequence(model_obs, eps=float(eps))
        except Exception:
            pass

    return chains if chains else None
