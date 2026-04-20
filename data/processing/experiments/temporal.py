"""Window-based baseline helpers for the EEG processing pipeline."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from experiments.models import (
    DEFAULT_REP_DIMS,
    evaluate_baseline_train_test,
    evaluate_prism_train_test,
    evaluate_trial_baseline,
)


DEFAULT_STIM_LOCKED_WINDOWS = (
    ("pre_far", -1500.0, -1000.0),
    ("pre_mid", -1000.0, -500.0),
    ("pre_near", -500.0, 0.0),
    ("post_early", 0.0, 500.0),
    ("post_mid", 500.0, 1000.0),
    ("post_late", 1000.0, 1500.0),
)
DEFAULT_SCALE_DURATIONS_MS = (250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0)
DEFAULT_CENTRE_SWEEP_DURATIONS_MS = (750.0, 1000.0, 1500.0, 2000.0, 3000.0)
DEFAULT_CENTRE_SWEEP_CENTRES_MS = (-750.0, -500.0, -250.0, 0.0, 250.0, 500.0, 750.0)
DEFAULT_CONTEXT_HISTORY_MS = (250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0)
DEFAULT_CONTEXT_TARGET_START_MS = 0.0
DEFAULT_CONTEXT_TARGET_END_MS = 1000.0
DEFAULT_SLIDING_HISTORY_MS = 500.0
DEFAULT_SLIDING_TARGET_DURATION_MS = 500.0
DEFAULT_SLIDING_STEP_MS = 250.0
DEFAULT_SLIDING_TARGET_START_MIN_MS = -1000.0
DEFAULT_SLIDING_TARGET_START_MAX_MS = 1000.0
DEFAULT_BASELINE_TRAIN_START_MS = -500.0
DEFAULT_BASELINE_TRAIN_END_MS = 0.0
DEFAULT_EXPLICIT_TRAIN_START_MS = -300.0
DEFAULT_EXPLICIT_TRAIN_END_MS = 0.0
DEFAULT_EXPLICIT_TEST_START_MS = 125.0
DEFAULT_EXPLICIT_TEST_END_MS = 375.0


def slice_trial_window(
    X: np.ndarray,
    times_ms: np.ndarray,
    start_ms: float,
    end_ms: float,
    *,
    require_full_window: bool = False,
    boundary_tolerance_ms: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"Expected trial with shape (time, channels), got {X.shape}")
    if times_ms.ndim != 1 or times_ms.shape[0] != X.shape[0]:
        raise ValueError(
            f"Expected times with shape ({X.shape[0]},), got {times_ms.shape}"
        )
    if start_ms >= end_ms:
        raise ValueError("Window start must be strictly less than window end")
    if boundary_tolerance_ms < 0.0:
        raise ValueError("boundary_tolerance_ms must be non-negative")

    if require_full_window:
        trial_start_ms = float(times_ms[0])
        trial_end_ms = float(times_ms[-1])
        if start_ms < (trial_start_ms - boundary_tolerance_ms) or end_ms > (
            trial_end_ms + boundary_tolerance_ms
        ):
            raise ValueError(
                "Requested window falls outside the observed trial support: "
                f"[{start_ms}, {end_ms}) ms vs [{trial_start_ms}, {trial_end_ms}] ms"
            )

    mask = (times_ms >= start_ms) & (times_ms < end_ms)
    if not np.any(mask):
        raise ValueError(
            f"Window [{start_ms}, {end_ms}) ms contains no samples for trial shape {X.shape}"
        )

    X_window = X[mask]
    times_window = times_ms[mask]
    if X_window.shape[0] < 3:
        raise ValueError(
            f"Window [{start_ms}, {end_ms}) ms is too short for VAR(1): {X_window.shape[0]} samples"
        )

    return X_window, times_window


def infer_boundary_tolerance_ms(times_ms: np.ndarray) -> float:
    times_ms = np.asarray(times_ms, dtype=float).reshape(-1)
    if times_ms.shape[0] < 2:
        return 0.0

    step_ms = float(np.median(np.diff(times_ms)))
    return max(0.0, step_ms)


def make_window_spec_error_rows(
    X: np.ndarray,
    rep_dims: Iterable[int],
    metadata: dict,
    error: str,
) -> list[dict]:
    return [
        {
            **metadata,
            "window_tmin_ms": np.nan,
            "window_tmax_ms": np.nan,
            "trial_n_time": int(X.shape[0]),
            "rep_dim": int(rep_dim),
            "n_time": np.nan,
            "n_channels": int(X.shape[1]),
            "train_len": np.nan,
            "test_len": np.nan,
            "train_pair_count": np.nan,
            "pred_mse_obs": np.nan,
            "pred_r2_obs": np.nan,
            "pred_mse_latent": np.nan,
            "pred_r2_latent": np.nan,
            "pred_nll_latent": np.nan,
            "error": error,
        }
        for rep_dim in rep_dims
    ]


def evaluate_trial_window_specs(
    X: np.ndarray,
    times_ms: np.ndarray,
    specs: Iterable[dict],
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    train_fraction: float = 0.7,
    require_full_window: bool = False,
    boundary_tolerance_ms: float = 0.0,
) -> list[dict]:
    X = np.asarray(X, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float).reshape(-1)
    rep_dims = tuple(int(dim) for dim in rep_dims)
    rows: list[dict] = []

    for spec in specs:
        start_ms = float(spec["start_ms"])
        end_ms = float(spec["end_ms"])
        metadata = {key: value for key, value in spec.items() if key not in {"start_ms", "end_ms"}}

        try:
            X_window, times_window = slice_trial_window(
                X,
                times_ms,
                start_ms,
                end_ms,
                require_full_window=require_full_window,
                boundary_tolerance_ms=boundary_tolerance_ms,
            )
            spec_rows = evaluate_trial_baseline(
                X_window,
                rep_dims=rep_dims,
                train_fraction=train_fraction,
            )
            for row in spec_rows:
                row.update(metadata)
                row["window_tmin_ms"] = float(times_window[0])
                row["window_tmax_ms"] = float(times_window[-1])
                row["trial_n_time"] = int(X.shape[0])
        except Exception as exc:
            spec_rows = make_window_spec_error_rows(
                X,
                rep_dims=rep_dims,
                metadata=metadata,
                error=str(exc),
            )

        rows.extend(spec_rows)

    return rows


def make_context_spec_error_rows(
    X: np.ndarray,
    rep_dims: Iterable[int],
    metadata: dict,
    error: str,
) -> list[dict]:
    return [
        {
            **metadata,
            "train_tmin_ms": np.nan,
            "train_tmax_ms": np.nan,
            "test_tmin_ms": np.nan,
            "test_tmax_ms": np.nan,
            "trial_n_time": int(X.shape[0]),
            "rep_dim": int(rep_dim),
            "n_time": np.nan,
            "n_channels": int(X.shape[1]),
            "train_len": np.nan,
            "test_len": np.nan,
            "train_pair_count": np.nan,
            "pred_mse_obs": np.nan,
            "pred_r2_obs": np.nan,
            "pred_mse_latent": np.nan,
            "pred_r2_latent": np.nan,
            "pred_nll_latent": np.nan,
            "error": error,
        }
        for rep_dim in rep_dims
    ]


def make_prism_context_spec_error_rows(
    X: np.ndarray,
    rep_dims: Iterable[int],
    projection_modes: Iterable[str],
    metadata: dict,
    error: str,
) -> list[dict]:
    rows = []
    for projection_mode in projection_modes:
        for rep_dim in rep_dims:
            rows.append(
                {
                    **metadata,
                    "train_tmin_ms": np.nan,
                    "train_tmax_ms": np.nan,
                    "test_tmin_ms": np.nan,
                    "test_tmax_ms": np.nan,
                    "trial_n_time": int(X.shape[0]),
                    "model_family": f"prism_{projection_mode}",
                    "projection_mode": projection_mode,
                    "rep_dim": int(rep_dim),
                    "latent_dim": int(rep_dim),
                    "macro_dim": int(rep_dim),
                    "n_time": np.nan,
                    "n_channels": int(X.shape[1]),
                    "train_len": np.nan,
                    "test_len": np.nan,
                    "pred_mse_obs": np.nan,
                    "pred_r2_obs": np.nan,
                    "pred_nll_obs": np.nan,
                    "macro_logloss": np.nan,
                    "n_macro_states": np.nan,
                    "psi_opt": np.nan,
                    "macro_builder": np.nan,
                    "iss_mode": np.nan,
                    "error": error,
                }
            )
    return rows


def build_window_specs(
    windows: tuple[tuple[str, float, float], ...] = DEFAULT_STIM_LOCKED_WINDOWS,
) -> tuple[dict, ...]:
    return tuple(
        {
            "start_ms": float(start_ms),
            "end_ms": float(end_ms),
            "window_name": window_name,
            "window_start_ms": float(start_ms),
            "window_end_ms": float(end_ms),
        }
        for window_name, start_ms, end_ms in windows
    )


def evaluate_trial_windows(
    X: np.ndarray,
    times_ms: np.ndarray,
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    train_fraction: float = 0.7,
    windows: tuple[tuple[str, float, float], ...] = DEFAULT_STIM_LOCKED_WINDOWS,
) -> list[dict]:
    specs = build_window_specs(windows=windows)
    return evaluate_trial_window_specs(
        X,
        times_ms,
        specs=specs,
        rep_dims=tuple(int(dim) for dim in rep_dims),
        train_fraction=train_fraction,
    )


def build_centred_scale_specs(
    durations_ms: Iterable[float] = DEFAULT_SCALE_DURATIONS_MS,
    center_ms: float = 0.0,
) -> tuple[dict, ...]:
    specs = []
    for duration_ms in durations_ms:
        duration_ms = float(duration_ms)
        if duration_ms <= 0.0:
            raise ValueError("All durations must be strictly positive")
        half = duration_ms / 2.0
        start_ms = center_ms - half
        end_ms = center_ms + half
        scale_name = f"dur_{int(round(duration_ms))}ms"
        specs.append(
            {
                "start_ms": float(start_ms),
                "end_ms": float(end_ms),
                "scale_name": scale_name,
                "duration_ms": float(duration_ms),
                "center_ms": float(center_ms),
                "window_start_ms": float(start_ms),
                "window_end_ms": float(end_ms),
            }
        )
    return tuple(specs)


def evaluate_trial_multiscale(
    X: np.ndarray,
    times_ms: np.ndarray,
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    train_fraction: float = 0.7,
    durations_ms: Iterable[float] = DEFAULT_SCALE_DURATIONS_MS,
    center_ms: float = 0.0,
) -> list[dict]:
    scale_specs = build_centred_scale_specs(durations_ms=durations_ms, center_ms=center_ms)
    return evaluate_trial_window_specs(
        np.asarray(X, dtype=float),
        times_ms,
        specs=scale_specs,
        rep_dims=tuple(int(dim) for dim in rep_dims),
        train_fraction=train_fraction,
    )


def format_centre_tag(centre_ms: float) -> str:
    rounded = int(round(centre_ms))
    if rounded < 0:
        return f"m{abs(rounded)}ms"
    if rounded > 0:
        return f"p{rounded}ms"
    return "0ms"


def build_centre_sweep_specs(
    centres_ms: Iterable[float] = DEFAULT_CENTRE_SWEEP_CENTRES_MS,
    durations_ms: Iterable[float] = DEFAULT_CENTRE_SWEEP_DURATIONS_MS,
) -> tuple[dict, ...]:
    specs = []
    for duration_ms in durations_ms:
        duration_ms = float(duration_ms)
        if duration_ms <= 0.0:
            raise ValueError("All durations must be strictly positive")

        half = duration_ms / 2.0
        for centre_ms in centres_ms:
            centre_ms = float(centre_ms)
            start_ms = centre_ms - half
            end_ms = centre_ms + half
            scale_name = (
                f"dur_{int(round(duration_ms))}ms_center_{format_centre_tag(centre_ms)}"
            )
            specs.append(
                {
                    "start_ms": float(start_ms),
                    "end_ms": float(end_ms),
                    "scale_name": scale_name,
                    "duration_ms": float(duration_ms),
                    "center_ms": float(centre_ms),
                    "window_start_ms": float(start_ms),
                    "window_end_ms": float(end_ms),
                }
            )

    return tuple(specs)


def build_context_sweep_specs(
    history_durations_ms: Iterable[float] = DEFAULT_CONTEXT_HISTORY_MS,
    target_start_ms: float = DEFAULT_CONTEXT_TARGET_START_MS,
    target_end_ms: float = DEFAULT_CONTEXT_TARGET_END_MS,
) -> tuple[dict, ...]:
    target_start_ms = float(target_start_ms)
    target_end_ms = float(target_end_ms)
    if target_start_ms >= target_end_ms:
        raise ValueError("target_start_ms must be strictly less than target_end_ms")

    specs = []
    for history_ms in history_durations_ms:
        history_ms = float(history_ms)
        if history_ms <= 0.0:
            raise ValueError("All history durations must be strictly positive")

        train_start_ms = target_start_ms - history_ms
        train_end_ms = target_start_ms
        context_name = f"history_{int(round(history_ms))}ms"
        specs.append(
            {
                "context_name": context_name,
                "history_ms": history_ms,
                "target_start_ms": target_start_ms,
                "target_end_ms": target_end_ms,
                "train_mode": "history_sweep",
                "baseline_duration_ms": np.nan,
                "train_start_ms": train_start_ms,
                "train_end_ms": train_end_ms,
                "test_start_ms": target_start_ms,
                "test_end_ms": target_end_ms,
            }
        )
    return tuple(specs)


def build_sliding_context_specs(
    history_ms: float = DEFAULT_SLIDING_HISTORY_MS,
    target_duration_ms: float = DEFAULT_SLIDING_TARGET_DURATION_MS,
    step_ms: float = DEFAULT_SLIDING_STEP_MS,
    target_start_min_ms: float = DEFAULT_SLIDING_TARGET_START_MIN_MS,
    target_start_max_ms: float = DEFAULT_SLIDING_TARGET_START_MAX_MS,
) -> tuple[dict, ...]:
    history_ms = float(history_ms)
    target_duration_ms = float(target_duration_ms)
    step_ms = float(step_ms)
    target_start_min_ms = float(target_start_min_ms)
    target_start_max_ms = float(target_start_max_ms)

    if history_ms <= 0.0:
        raise ValueError("history_ms must be strictly positive")
    if target_duration_ms <= 0.0:
        raise ValueError("target_duration_ms must be strictly positive")
    if step_ms <= 0.0:
        raise ValueError("step_ms must be strictly positive")
    if target_start_min_ms > target_start_max_ms:
        raise ValueError("target_start_min_ms must be less than or equal to target_start_max_ms")

    n_steps = int(np.floor((target_start_max_ms - target_start_min_ms) / step_ms)) + 1
    target_starts = target_start_min_ms + step_ms * np.arange(n_steps, dtype=float)
    specs = []

    for target_start_ms in target_starts:
        target_end_ms = target_start_ms + target_duration_ms
        target_center_ms = target_start_ms + 0.5 * target_duration_ms
        specs.append(
            {
                "window_name": f"target_{int(round(target_start_ms))}ms",
                "history_ms": history_ms,
                "target_duration_ms": target_duration_ms,
                "step_ms": step_ms,
                "train_mode": "rolling_history",
                "baseline_duration_ms": np.nan,
                "target_start_ms": float(target_start_ms),
                "target_end_ms": float(target_end_ms),
                "target_center_ms": float(target_center_ms),
                "train_start_ms": float(target_start_ms - history_ms),
                "train_end_ms": float(target_start_ms),
                "test_start_ms": float(target_start_ms),
                "test_end_ms": float(target_end_ms),
            }
        )

    return tuple(specs)


def build_baseline_sliding_specs(
    train_start_ms: float = DEFAULT_BASELINE_TRAIN_START_MS,
    train_end_ms: float = DEFAULT_BASELINE_TRAIN_END_MS,
    target_duration_ms: float = DEFAULT_SLIDING_TARGET_DURATION_MS,
    step_ms: float = DEFAULT_SLIDING_STEP_MS,
    target_start_min_ms: float = DEFAULT_SLIDING_TARGET_START_MIN_MS,
    target_start_max_ms: float = DEFAULT_SLIDING_TARGET_START_MAX_MS,
) -> tuple[dict, ...]:
    train_start_ms = float(train_start_ms)
    train_end_ms = float(train_end_ms)
    target_duration_ms = float(target_duration_ms)
    step_ms = float(step_ms)
    target_start_min_ms = float(target_start_min_ms)
    target_start_max_ms = float(target_start_max_ms)

    if train_start_ms >= train_end_ms:
        raise ValueError("train_start_ms must be strictly less than train_end_ms")
    if target_duration_ms <= 0.0:
        raise ValueError("target_duration_ms must be strictly positive")
    if step_ms <= 0.0:
        raise ValueError("step_ms must be strictly positive")
    if target_start_min_ms > target_start_max_ms:
        raise ValueError("target_start_min_ms must be less than or equal to target_start_max_ms")

    baseline_duration_ms = train_end_ms - train_start_ms
    n_steps = int(np.floor((target_start_max_ms - target_start_min_ms) / step_ms)) + 1
    target_starts = target_start_min_ms + step_ms * np.arange(n_steps, dtype=float)
    specs = []

    for target_start_ms in target_starts:
        target_end_ms = target_start_ms + target_duration_ms
        target_center_ms = target_start_ms + 0.5 * target_duration_ms
        specs.append(
            {
                "window_name": f"target_{int(round(target_start_ms))}ms",
                "history_ms": np.nan,
                "target_duration_ms": target_duration_ms,
                "step_ms": step_ms,
                "train_mode": "fixed_baseline",
                "baseline_duration_ms": float(baseline_duration_ms),
                "target_start_ms": float(target_start_ms),
                "target_end_ms": float(target_end_ms),
                "target_center_ms": float(target_center_ms),
                "train_start_ms": float(train_start_ms),
                "train_end_ms": float(train_end_ms),
                "test_start_ms": float(target_start_ms),
                "test_end_ms": float(target_end_ms),
            }
        )

    return tuple(specs)


def build_explicit_train_test_spec(
    *,
    train_start_ms: float = DEFAULT_EXPLICIT_TRAIN_START_MS,
    train_end_ms: float = DEFAULT_EXPLICIT_TRAIN_END_MS,
    test_start_ms: float = DEFAULT_EXPLICIT_TEST_START_MS,
    test_end_ms: float = DEFAULT_EXPLICIT_TEST_END_MS,
) -> tuple[dict]:
    train_start_ms = float(train_start_ms)
    train_end_ms = float(train_end_ms)
    test_start_ms = float(test_start_ms)
    test_end_ms = float(test_end_ms)

    if train_start_ms >= train_end_ms:
        raise ValueError("train_start_ms must be strictly less than train_end_ms")
    if test_start_ms >= test_end_ms:
        raise ValueError("test_start_ms must be strictly less than test_end_ms")

    train_duration_ms = train_end_ms - train_start_ms
    test_duration_ms = test_end_ms - test_start_ms
    train_mode = "fixed_baseline" if train_end_ms <= 0.0 else "fixed_window"

    return (
        {
            "window_name": (
                f"train_{int(round(train_start_ms))}_{int(round(train_end_ms))}_"
                f"test_{int(round(test_start_ms))}_{int(round(test_end_ms))}"
            ),
            "train_mode": train_mode,
            "history_ms": np.nan,
            "baseline_duration_ms": float(train_duration_ms)
            if train_mode == "fixed_baseline"
            else np.nan,
            "train_duration_ms": float(train_duration_ms),
            "target_duration_ms": float(test_duration_ms),
            "target_center_ms": float(test_start_ms + 0.5 * test_duration_ms),
            "train_start_ms": float(train_start_ms),
            "train_end_ms": float(train_end_ms),
            "test_start_ms": float(test_start_ms),
            "test_end_ms": float(test_end_ms),
        },
    )


def evaluate_trial_train_test_specs(
    X: np.ndarray,
    times_ms: np.ndarray,
    *,
    rep_dims: Iterable[int],
    specs: Iterable[dict],
    boundary_tolerance_ms: float,
    max_train_pairs: int | None = None,
) -> list[dict]:
    X = np.asarray(X, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float).reshape(-1)
    rep_dims = tuple(int(dim) for dim in rep_dims)

    rows: list[dict] = []
    for spec in specs:
        metadata = dict(spec)

        try:
            train_X, train_times = slice_trial_window(
                X,
                times_ms,
                spec["train_start_ms"],
                spec["train_end_ms"],
                require_full_window=True,
                boundary_tolerance_ms=boundary_tolerance_ms,
            )
            test_X, test_times = slice_trial_window(
                X,
                times_ms,
                spec["test_start_ms"],
                spec["test_end_ms"],
                require_full_window=True,
                boundary_tolerance_ms=boundary_tolerance_ms,
            )
            spec_rows = evaluate_baseline_train_test(
                train_X,
                test_X,
                rep_dims=rep_dims,
                n_time=int(train_X.shape[0] + test_X.shape[0]),
                max_train_pairs=max_train_pairs,
            )
            for row in spec_rows:
                row.update(metadata)
                row["train_tmin_ms"] = float(train_times[0])
                row["train_tmax_ms"] = float(train_times[-1])
                row["test_tmin_ms"] = float(test_times[0])
                row["test_tmax_ms"] = float(test_times[-1])
                row["trial_n_time"] = int(X.shape[0])
        except Exception as exc:
            spec_rows = make_context_spec_error_rows(
                X,
                rep_dims=rep_dims,
                metadata=metadata,
                error=str(exc),
            )

        rows.extend(spec_rows)

    return rows


def evaluate_trial_prism_train_test_specs(
    X: np.ndarray,
    times_ms: np.ndarray,
    *,
    rep_dims: Iterable[int],
    projection_modes: Iterable[str],
    specs: Iterable[dict],
    boundary_tolerance_ms: float,
    macro_builder: str = "hierarchical_complete",
    macro_eps: float = 0.25,
    macro_bins: int = 3,
    em_iters: int = 20,
    em_tol: float = 1e-3,
    em_ridge: float = 1e-6,
    iss_mode: str = "steady_state",
    allow_time_varying_fallback: bool = False,
    random_state: int = 0,
) -> list[dict]:
    X = np.asarray(X, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float).reshape(-1)
    rep_dims = tuple(int(dim) for dim in rep_dims)
    projection_modes = tuple(str(mode) for mode in projection_modes)

    rows: list[dict] = []
    for spec_idx, spec in enumerate(specs):
        metadata = dict(spec)

        try:
            train_X, train_times = slice_trial_window(
                X,
                times_ms,
                spec["train_start_ms"],
                spec["train_end_ms"],
                require_full_window=True,
                boundary_tolerance_ms=boundary_tolerance_ms,
            )
            test_X, test_times = slice_trial_window(
                X,
                times_ms,
                spec["test_start_ms"],
                spec["test_end_ms"],
                require_full_window=True,
                boundary_tolerance_ms=boundary_tolerance_ms,
            )
            spec_rows = evaluate_prism_train_test(
                train_X,
                test_X,
                rep_dims=rep_dims,
                projection_modes=projection_modes,
                n_time=int(train_X.shape[0] + test_X.shape[0]),
                macro_builder=macro_builder,
                macro_eps=macro_eps,
                macro_bins=macro_bins,
                em_iters=em_iters,
                em_tol=em_tol,
                em_ridge=em_ridge,
                iss_mode=iss_mode,
                allow_time_varying_fallback=allow_time_varying_fallback,
                random_state=int(random_state + spec_idx),
            )
            for row in spec_rows:
                row.update(metadata)
                row["train_tmin_ms"] = float(train_times[0])
                row["train_tmax_ms"] = float(train_times[-1])
                row["test_tmin_ms"] = float(test_times[0])
                row["test_tmax_ms"] = float(test_times[-1])
                row["trial_n_time"] = int(X.shape[0])
        except Exception as exc:
            spec_rows = make_prism_context_spec_error_rows(
                X,
                rep_dims=rep_dims,
                projection_modes=projection_modes,
                metadata=metadata,
                error=str(exc),
            )

        rows.extend(spec_rows)

    return rows


def evaluate_trial_centre_sweep(
    X: np.ndarray,
    times_ms: np.ndarray,
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    train_fraction: float = 0.7,
    centres_ms: Iterable[float] = DEFAULT_CENTRE_SWEEP_CENTRES_MS,
    durations_ms: Iterable[float] = DEFAULT_CENTRE_SWEEP_DURATIONS_MS,
    boundary_tolerance_ms: float | None = None,
    sweep_specs: tuple[dict, ...] | None = None,
) -> list[dict]:
    X = np.asarray(X, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float).reshape(-1)

    if boundary_tolerance_ms is None:
        boundary_tolerance_ms = infer_boundary_tolerance_ms(times_ms)

    if sweep_specs is None:
        sweep_specs = build_centre_sweep_specs(
            centres_ms=centres_ms,
            durations_ms=durations_ms,
        )

    return evaluate_trial_window_specs(
        X,
        times_ms,
        specs=sweep_specs,
        rep_dims=tuple(int(dim) for dim in rep_dims),
        train_fraction=train_fraction,
        require_full_window=True,
        boundary_tolerance_ms=boundary_tolerance_ms,
    )


def evaluate_trial_context_sweep(
    X: np.ndarray,
    times_ms: np.ndarray,
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    history_durations_ms: Iterable[float] = DEFAULT_CONTEXT_HISTORY_MS,
    target_start_ms: float = DEFAULT_CONTEXT_TARGET_START_MS,
    target_end_ms: float = DEFAULT_CONTEXT_TARGET_END_MS,
    max_train_pairs: int | None = None,
    boundary_tolerance_ms: float | None = None,
    context_specs: tuple[dict, ...] | None = None,
) -> list[dict]:
    X = np.asarray(X, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float).reshape(-1)
    rep_dims = tuple(int(dim) for dim in rep_dims)

    if boundary_tolerance_ms is None:
        boundary_tolerance_ms = infer_boundary_tolerance_ms(times_ms)

    if context_specs is None:
        context_specs = build_context_sweep_specs(
            history_durations_ms=history_durations_ms,
            target_start_ms=target_start_ms,
            target_end_ms=target_end_ms,
        )

    return evaluate_trial_train_test_specs(
        X,
        times_ms,
        rep_dims=rep_dims,
        specs=context_specs,
        boundary_tolerance_ms=boundary_tolerance_ms,
        max_train_pairs=max_train_pairs,
    )


def evaluate_trial_sliding_context(
    X: np.ndarray,
    times_ms: np.ndarray,
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    history_ms: float = DEFAULT_SLIDING_HISTORY_MS,
    target_duration_ms: float = DEFAULT_SLIDING_TARGET_DURATION_MS,
    step_ms: float = DEFAULT_SLIDING_STEP_MS,
    target_start_min_ms: float = DEFAULT_SLIDING_TARGET_START_MIN_MS,
    target_start_max_ms: float = DEFAULT_SLIDING_TARGET_START_MAX_MS,
    boundary_tolerance_ms: float | None = None,
    sliding_specs: tuple[dict, ...] | None = None,
    max_train_pairs: int | None = None,
) -> list[dict]:
    X = np.asarray(X, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float).reshape(-1)
    rep_dims = tuple(int(dim) for dim in rep_dims)

    if boundary_tolerance_ms is None:
        boundary_tolerance_ms = infer_boundary_tolerance_ms(times_ms)

    if sliding_specs is None:
        sliding_specs = build_sliding_context_specs(
            history_ms=history_ms,
            target_duration_ms=target_duration_ms,
            step_ms=step_ms,
            target_start_min_ms=target_start_min_ms,
            target_start_max_ms=target_start_max_ms,
        )

    return evaluate_trial_train_test_specs(
        X,
        times_ms,
        rep_dims=rep_dims,
        specs=sliding_specs,
        boundary_tolerance_ms=boundary_tolerance_ms,
        max_train_pairs=max_train_pairs,
    )


def evaluate_trial_baseline_sliding(
    X: np.ndarray,
    times_ms: np.ndarray,
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    train_start_ms: float = DEFAULT_BASELINE_TRAIN_START_MS,
    train_end_ms: float = DEFAULT_BASELINE_TRAIN_END_MS,
    target_duration_ms: float = DEFAULT_SLIDING_TARGET_DURATION_MS,
    step_ms: float = DEFAULT_SLIDING_STEP_MS,
    target_start_min_ms: float = DEFAULT_SLIDING_TARGET_START_MIN_MS,
    target_start_max_ms: float = DEFAULT_SLIDING_TARGET_START_MAX_MS,
    boundary_tolerance_ms: float | None = None,
    sliding_specs: tuple[dict, ...] | None = None,
    max_train_pairs: int | None = None,
) -> list[dict]:
    X = np.asarray(X, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float).reshape(-1)
    rep_dims = tuple(int(dim) for dim in rep_dims)

    if boundary_tolerance_ms is None:
        boundary_tolerance_ms = infer_boundary_tolerance_ms(times_ms)

    if sliding_specs is None:
        sliding_specs = build_baseline_sliding_specs(
            train_start_ms=train_start_ms,
            train_end_ms=train_end_ms,
            target_duration_ms=target_duration_ms,
            step_ms=step_ms,
            target_start_min_ms=target_start_min_ms,
            target_start_max_ms=target_start_max_ms,
        )

    return evaluate_trial_train_test_specs(
        X,
        times_ms,
        rep_dims=rep_dims,
        specs=sliding_specs,
        boundary_tolerance_ms=boundary_tolerance_ms,
        max_train_pairs=max_train_pairs,
    )


def evaluate_trial_baseline_window(
    X: np.ndarray,
    times_ms: np.ndarray,
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    train_start_ms: float = DEFAULT_EXPLICIT_TRAIN_START_MS,
    train_end_ms: float = DEFAULT_EXPLICIT_TRAIN_END_MS,
    test_start_ms: float = DEFAULT_EXPLICIT_TEST_START_MS,
    test_end_ms: float = DEFAULT_EXPLICIT_TEST_END_MS,
    boundary_tolerance_ms: float | None = None,
    max_train_pairs: int | None = None,
) -> list[dict]:
    X = np.asarray(X, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float).reshape(-1)
    rep_dims = tuple(int(dim) for dim in rep_dims)

    if boundary_tolerance_ms is None:
        boundary_tolerance_ms = infer_boundary_tolerance_ms(times_ms)

    specs = build_explicit_train_test_spec(
        train_start_ms=train_start_ms,
        train_end_ms=train_end_ms,
        test_start_ms=test_start_ms,
        test_end_ms=test_end_ms,
    )
    return evaluate_trial_train_test_specs(
        X,
        times_ms,
        rep_dims=rep_dims,
        specs=specs,
        boundary_tolerance_ms=boundary_tolerance_ms,
        max_train_pairs=max_train_pairs,
    )


def evaluate_trial_prism_window(
    X: np.ndarray,
    times_ms: np.ndarray,
    *,
    rep_dims: Iterable[int] = DEFAULT_REP_DIMS,
    projection_modes: Iterable[str],
    train_start_ms: float = DEFAULT_EXPLICIT_TRAIN_START_MS,
    train_end_ms: float = DEFAULT_EXPLICIT_TRAIN_END_MS,
    test_start_ms: float = DEFAULT_EXPLICIT_TEST_START_MS,
    test_end_ms: float = DEFAULT_EXPLICIT_TEST_END_MS,
    boundary_tolerance_ms: float | None = None,
    macro_builder: str = "hierarchical_complete",
    macro_eps: float = 0.25,
    macro_bins: int = 3,
    em_iters: int = 20,
    em_tol: float = 1e-3,
    em_ridge: float = 1e-6,
    iss_mode: str = "steady_state",
    allow_time_varying_fallback: bool = False,
    random_state: int = 0,
) -> list[dict]:
    X = np.asarray(X, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float).reshape(-1)
    rep_dims = tuple(int(dim) for dim in rep_dims)

    if boundary_tolerance_ms is None:
        boundary_tolerance_ms = infer_boundary_tolerance_ms(times_ms)

    specs = build_explicit_train_test_spec(
        train_start_ms=train_start_ms,
        train_end_ms=train_end_ms,
        test_start_ms=test_start_ms,
        test_end_ms=test_end_ms,
    )
    return evaluate_trial_prism_train_test_specs(
        X,
        times_ms,
        rep_dims=rep_dims,
        projection_modes=projection_modes,
        specs=specs,
        boundary_tolerance_ms=boundary_tolerance_ms,
        macro_builder=macro_builder,
        macro_eps=macro_eps,
        macro_bins=macro_bins,
        em_iters=em_iters,
        em_tol=em_tol,
        em_ridge=em_ridge,
        iss_mode=iss_mode,
        allow_time_varying_fallback=allow_time_varying_fallback,
        random_state=random_state,
    )
