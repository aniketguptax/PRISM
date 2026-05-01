"""Registry of runnable EEG processing experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from eegprep import load_and_prepare_subject, load_subject_channel_labels
from experiments.models import (
    DEFAULT_CONTROL_KINDS,
    DEFAULT_PRISM_PROJECTION_MODES,
    DEFAULT_REP_DIMS,
    evaluate_trial_baseline,
    evaluate_trial_controls,
    evaluate_trial_iss,
    evaluate_trial_prism,
)
from experiments.spatial import (
    DEFAULT_REGION_ORDER,
    RegionGroup,
    add_region_metadata,
    build_control_region_groups,
    rep_dims_for_group,
    select_region_groups,
)
from experiments.temporal import (
    DEFAULT_BASELINE_TRAIN_END_MS,
    DEFAULT_BASELINE_TRAIN_START_MS,
    DEFAULT_CONTEXT_HISTORY_MS,
    DEFAULT_CONTEXT_TARGET_END_MS,
    DEFAULT_CONTEXT_TARGET_START_MS,
    DEFAULT_CENTRE_SWEEP_CENTRES_MS,
    DEFAULT_CENTRE_SWEEP_DURATIONS_MS,
    DEFAULT_EXPLICIT_TEST_END_MS,
    DEFAULT_EXPLICIT_TEST_START_MS,
    DEFAULT_EXPLICIT_TRAIN_END_MS,
    DEFAULT_EXPLICIT_TRAIN_START_MS,
    DEFAULT_SCALE_DURATIONS_MS,
    DEFAULT_SLIDING_HISTORY_MS,
    DEFAULT_SLIDING_STEP_MS,
    DEFAULT_SLIDING_TARGET_DURATION_MS,
    DEFAULT_SLIDING_TARGET_START_MAX_MS,
    DEFAULT_SLIDING_TARGET_START_MIN_MS,
    DEFAULT_STIM_LOCKED_WINDOWS,
    build_centre_sweep_specs,
    evaluate_trial_baseline_sliding,
    evaluate_trial_baseline_window,
    evaluate_trial_iss_baseline_sliding,
    evaluate_trial_centre_sweep,
    evaluate_trial_context_sweep,
    evaluate_trial_multiscale,
    evaluate_trial_prism_window,
    evaluate_trial_sliding_context,
    evaluate_trial_windows,
)
from framework.runtime import extract_subject_id, load_subject_export


ParserMutator = Callable[[argparse.ArgumentParser], None]
ConfigBuilder = Callable[[argparse.Namespace], dict]
EvaluateFactory = Callable[[dict, Path], Callable[[np.ndarray, np.ndarray], list[dict]]]
DescribeRun = Callable[[pd.DataFrame, str], str]
LoaderFn = Callable[[Path], tuple[np.ndarray, float, np.ndarray]]


COMMON_OUTPUT_COLUMNS = (
    "subject",
    "trial_idx",
    "rep_dim",
    "n_time",
    "n_channels",
    "train_len",
    "test_len",
    "train_pair_count",
    "pred_mse_obs",
    "pred_r2_obs",
    "pred_mse_latent",
    "pred_r2_latent",
    "pred_nll_latent",
    "sfreq",
    "tmin_ms",
    "tmax_ms",
    "error",
)
WINDOW_OUTPUT_COLUMNS = (
    "subject",
    "trial_idx",
    "window_name",
    "window_start_ms",
    "window_end_ms",
    "window_tmin_ms",
    "window_tmax_ms",
    "trial_n_time",
    "rep_dim",
    "n_time",
    "n_channels",
    "train_len",
    "test_len",
    "pred_mse_obs",
    "pred_r2_obs",
    "pred_mse_latent",
    "pred_r2_latent",
    "pred_nll_latent",
    "sfreq",
    "trial_tmin_ms",
    "trial_tmax_ms",
    "error",
)
SCALE_OUTPUT_COLUMNS = (
    "subject",
    "trial_idx",
    "scale_name",
    "duration_ms",
    "center_ms",
    "window_start_ms",
    "window_end_ms",
    "window_tmin_ms",
    "window_tmax_ms",
    "trial_n_time",
    "rep_dim",
    "n_time",
    "n_channels",
    "train_len",
    "test_len",
    "pred_mse_obs",
    "pred_r2_obs",
    "pred_mse_latent",
    "pred_r2_latent",
    "pred_nll_latent",
    "sfreq",
    "trial_tmin_ms",
    "trial_tmax_ms",
    "error",
)
CONTEXT_OUTPUT_COLUMNS = (
    "subject",
    "trial_idx",
    "context_name",
    "history_ms",
    "target_start_ms",
    "target_end_ms",
    "train_start_ms",
    "train_end_ms",
    "test_start_ms",
    "test_end_ms",
    "train_tmin_ms",
    "train_tmax_ms",
    "test_tmin_ms",
    "test_tmax_ms",
    "trial_n_time",
    "rep_dim",
    "n_time",
    "n_channels",
    "train_len",
    "test_len",
    "train_pair_count",
    "pred_mse_obs",
    "pred_r2_obs",
    "pred_mse_latent",
    "pred_r2_latent",
    "pred_nll_latent",
    "sfreq",
    "trial_tmin_ms",
    "trial_tmax_ms",
    "error",
)
TRAIN_TEST_WINDOW_OUTPUT_COLUMNS = (
    "subject",
    "trial_idx",
    "window_name",
    "train_mode",
    "history_ms",
    "baseline_duration_ms",
    "train_duration_ms",
    "target_duration_ms",
    "target_center_ms",
    "train_start_ms",
    "train_end_ms",
    "test_start_ms",
    "test_end_ms",
    "train_tmin_ms",
    "train_tmax_ms",
    "test_tmin_ms",
    "test_tmax_ms",
    "trial_n_time",
    "rep_dim",
    "n_time",
    "n_channels",
    "train_len",
    "test_len",
    "train_pair_count",
    "pred_mse_obs",
    "pred_r2_obs",
    "pred_mse_latent",
    "pred_r2_latent",
    "pred_nll_latent",
    "sfreq",
    "trial_tmin_ms",
    "trial_tmax_ms",
    "error",
)
PRISM_WINDOW_OUTPUT_COLUMNS = (
    "subject",
    "trial_idx",
    "window_name",
    "train_mode",
    "history_ms",
    "baseline_duration_ms",
    "train_duration_ms",
    "target_duration_ms",
    "target_center_ms",
    "train_start_ms",
    "train_end_ms",
    "test_start_ms",
    "test_end_ms",
    "train_tmin_ms",
    "train_tmax_ms",
    "test_tmin_ms",
    "test_tmax_ms",
    "trial_n_time",
    "model_family",
    "projection_mode",
    "rep_dim",
    "latent_dim",
    "macro_dim",
    "n_time",
    "n_channels",
    "train_len",
    "test_len",
    "pred_mse_obs",
    "pred_r2_obs",
    "pred_nll_obs",
    "macro_logloss",
    "n_macro_states",
    "psi_opt",
    "macro_builder",
    "iss_mode",
    "sfreq",
    "trial_tmin_ms",
    "trial_tmax_ms",
    "error",
)
PRISM_REGION_WINDOW_OUTPUT_COLUMNS = (
    "subject",
    "trial_idx",
    "region_name",
    "n_region_channels",
    "window_name",
    "train_mode",
    "history_ms",
    "baseline_duration_ms",
    "train_duration_ms",
    "target_duration_ms",
    "target_center_ms",
    "train_start_ms",
    "train_end_ms",
    "test_start_ms",
    "test_end_ms",
    "train_tmin_ms",
    "train_tmax_ms",
    "test_tmin_ms",
    "test_tmax_ms",
    "trial_n_time",
    "model_family",
    "projection_mode",
    "rep_dim",
    "latent_dim",
    "macro_dim",
    "n_time",
    "n_channels",
    "train_len",
    "test_len",
    "pred_mse_obs",
    "pred_r2_obs",
    "pred_nll_obs",
    "macro_logloss",
    "n_macro_states",
    "psi_opt",
    "macro_builder",
    "iss_mode",
    "sfreq",
    "trial_tmin_ms",
    "trial_tmax_ms",
    "error",
)
REGION_SLIDING_OUTPUT_COLUMNS = (
    "subject",
    "trial_idx",
    "region_name",
    "group_kind",
    "matched_region_name",
    "control_draw_idx",
    "n_region_channels",
    "window_name",
    "train_mode",
    "history_ms",
    "baseline_duration_ms",
    "target_duration_ms",
    "step_ms",
    "target_start_ms",
    "target_end_ms",
    "target_center_ms",
    "train_start_ms",
    "train_end_ms",
    "test_start_ms",
    "test_end_ms",
    "train_tmin_ms",
    "train_tmax_ms",
    "test_tmin_ms",
    "test_tmax_ms",
    "trial_n_time",
    "rep_dim",
    "n_time",
    "n_channels",
    "train_len",
    "test_len",
    "train_pair_count",
    "pred_mse_obs",
    "pred_r2_obs",
    "pred_mse_latent",
    "pred_r2_latent",
    "pred_nll_latent",
    "sfreq",
    "trial_tmin_ms",
    "trial_tmax_ms",
    "error",
)
CONTROL_OUTPUT_COLUMNS = (
    "subject",
    "trial_idx",
    "control_kind",
    "control_seed",
    "rep_dim",
    "n_time",
    "n_channels",
    "train_len",
    "test_len",
    "pred_mse_obs",
    "pred_r2_obs",
    "pred_mse_latent",
    "pred_r2_latent",
    "pred_nll_latent",
    "sfreq",
    "tmin_ms",
    "tmax_ms",
    "error",
)
PRISM_OUTPUT_COLUMNS = (
    "subject",
    "trial_idx",
    "model_family",
    "projection_mode",
    "rep_dim",
    "latent_dim",
    "macro_dim",
    "n_time",
    "n_channels",
    "train_len",
    "test_len",
    "pred_mse_obs",
    "pred_r2_obs",
    "pred_nll_obs",
    "macro_logloss",
    "n_macro_states",
    "psi_opt",
    "macro_builder",
    "iss_mode",
    "sfreq",
    "tmin_ms",
    "tmax_ms",
    "error",
)
TRIAL_METRIC_OUTPUT_COLUMNS = (
    "subject",
    "trial_idx",
    "n_time",
    "n_channels",
    "mean_abs",
    "var",
    "mean_l2",
    "sfreq",
    "tmin_ms",
    "tmax_ms",
    "error",
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    default_outdir: str
    combined_filename: str
    output_suffix: str
    output_columns: tuple[str, ...]
    evaluate_trial_factory: EvaluateFactory
    sort_columns: tuple[str, ...] | None = None
    load_subject_fn: LoaderFn = load_subject_export
    time_bounds_keys: tuple[str, str] = ("trial_tmin_ms", "trial_tmax_ms")
    progress_every: int = 100
    train_fraction_help: str | None = None
    rep_dims_default: tuple[int, ...] | None = None
    rep_dims_help: str = "Representation dimensions to evaluate."
    add_arguments: ParserMutator | None = None
    config_from_args: ConfigBuilder | None = None
    describe_run: DescribeRun | None = None
    aliases: tuple[str, ...] = ()


def basic_trial_metrics(X: np.ndarray) -> dict:
    return {
        "n_time": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "mean_abs": float(np.mean(np.abs(X))),
        "var": float(np.var(X)),
        "mean_l2": float(np.mean(np.linalg.norm(X, axis=1))),
        "error": "",
    }


def subject_seed(key: str, *, base: int = 0) -> int:
    return int(base + sum(ord(char) for char in key))


def load_requested_region_groups(
    subject: str,
    config: dict,
) -> tuple[list[str], tuple[RegionGroup, ...]]:
    channel_labels = load_subject_channel_labels(
        subject,
        derivatives_dir=config["derivatives_dir"],
    )
    selected_groups = select_region_groups(channel_labels, config["regions"])
    return channel_labels, selected_groups


def add_multiscale_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--durations-ms",
        nargs="+",
        type=float,
        default=list(DEFAULT_SCALE_DURATIONS_MS),
        help="Centred window durations in milliseconds.",
    )
    parser.add_argument(
        "--centre-ms",
        "--center-ms",
        type=float,
        default=0.0,
        help="Centre of the multiscale window sweep in milliseconds.",
    )


def add_centre_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--durations-ms",
        nargs="+",
        type=float,
        default=list(DEFAULT_CENTRE_SWEEP_DURATIONS_MS),
        help="Window durations in milliseconds.",
    )
    parser.add_argument(
        "--centres-ms",
        "--centers-ms",
        nargs="+",
        type=float,
        default=list(DEFAULT_CENTRE_SWEEP_CENTRES_MS),
        help="Stimulus-relative window centres in milliseconds.",
    )


def add_context_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--history-durations-ms",
        nargs="+",
        type=float,
        default=list(DEFAULT_CONTEXT_HISTORY_MS),
        help="History durations in milliseconds, ending at the target start.",
    )
    parser.add_argument(
        "--target-start-ms",
        type=float,
        default=DEFAULT_CONTEXT_TARGET_START_MS,
        help="Start of the fixed prediction target window in milliseconds.",
    )
    parser.add_argument(
        "--target-end-ms",
        type=float,
        default=DEFAULT_CONTEXT_TARGET_END_MS,
        help="End of the fixed prediction target window in milliseconds.",
    )
    parser.add_argument(
        "--max-train-pairs",
        type=int,
        default=None,
        help="If set, fit the VAR baseline with at most this many evenly spaced training pairs.",
    )


def add_train_test_window_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-start-ms",
        type=float,
        default=DEFAULT_EXPLICIT_TRAIN_START_MS,
        help="Start of the fixed training window in milliseconds.",
    )
    parser.add_argument(
        "--train-end-ms",
        type=float,
        default=DEFAULT_EXPLICIT_TRAIN_END_MS,
        help="End of the fixed training window in milliseconds.",
    )
    parser.add_argument(
        "--test-start-ms",
        type=float,
        default=DEFAULT_EXPLICIT_TEST_START_MS,
        help="Start of the fixed prediction window in milliseconds.",
    )
    parser.add_argument(
        "--test-end-ms",
        type=float,
        default=DEFAULT_EXPLICIT_TEST_END_MS,
        help="End of the fixed prediction window in milliseconds.",
    )
    parser.add_argument(
        "--max-train-pairs",
        type=int,
        default=None,
        help="If set, fit the VAR baseline with at most this many evenly spaced training pairs.",
    )


def add_prism_window_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-start-ms",
        type=float,
        default=DEFAULT_EXPLICIT_TRAIN_START_MS,
        help="Start of the fixed training window in milliseconds.",
    )
    parser.add_argument(
        "--train-end-ms",
        type=float,
        default=DEFAULT_EXPLICIT_TRAIN_END_MS,
        help="End of the fixed training window in milliseconds.",
    )
    parser.add_argument(
        "--test-start-ms",
        type=float,
        default=DEFAULT_EXPLICIT_TEST_START_MS,
        help="Start of the fixed prediction window in milliseconds.",
    )
    parser.add_argument(
        "--test-end-ms",
        type=float,
        default=DEFAULT_EXPLICIT_TEST_END_MS,
        help="End of the fixed prediction window in milliseconds.",
    )
    add_prism_arguments(parser)


def add_prism_region_window_arguments(parser: argparse.ArgumentParser) -> None:
    add_prism_window_arguments(parser)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGION_ORDER),
        help=(
            "Channel groups to evaluate. Defaults to all_channels plus five scalp regions: "
            "frontal, central, temporal, parietal, occipital."
        ),
    )
    parser.add_argument(
        "--derivatives-dir",
        default="./data/ds001785/derivatives/eegprep",
        help="Directory containing the subject .set files used to recover exported channel labels.",
    )


def add_region_sliding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--history-ms",
        type=float,
        default=DEFAULT_SLIDING_HISTORY_MS,
        help="Length of the history window in milliseconds.",
    )
    parser.add_argument(
        "--target-duration-ms",
        type=float,
        default=DEFAULT_SLIDING_TARGET_DURATION_MS,
        help="Length of the fixed prediction target window in milliseconds.",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=DEFAULT_SLIDING_STEP_MS,
        help="Step size for sliding the target window in milliseconds.",
    )
    parser.add_argument(
        "--target-start-min-ms",
        type=float,
        default=DEFAULT_SLIDING_TARGET_START_MIN_MS,
        help="Earliest target-window start in milliseconds.",
    )
    parser.add_argument(
        "--target-start-max-ms",
        type=float,
        default=DEFAULT_SLIDING_TARGET_START_MAX_MS,
        help="Latest target-window start in milliseconds.",
    )
    parser.add_argument(
        "--train-start-ms",
        type=float,
        default=None,
        help=(
            "Optional fixed training-window start in milliseconds. When set together with "
            "--train-end-ms, the same baseline segment is used to fit every sliding window."
        ),
    )
    parser.add_argument(
        "--train-end-ms",
        type=float,
        default=None,
        help=(
            "Optional fixed training-window end in milliseconds. Typical baseline choices are "
            f"{int(DEFAULT_BASELINE_TRAIN_START_MS)} to {int(DEFAULT_BASELINE_TRAIN_END_MS)} ms."
        ),
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGION_ORDER),
        help=(
            "Channel groups to evaluate. Defaults to all_channels plus five scalp regions: "
            "frontal, central, temporal, parietal, occipital."
        ),
    )
    parser.add_argument(
        "--derivatives-dir",
        default="./data/ds001785/derivatives/eegprep",
        help="Directory containing the subject .set files used to recover exported channel labels.",
    )
    parser.add_argument(
        "--random-control-draws",
        type=int,
        default=0,
        help=(
            "Number of deterministic size-matched random channel groups to draw for each named "
            "region. Controls are sampled from the remaining cortical channels."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Base random seed used to derive deterministic size-matched channel controls.",
    )


def add_control_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--control-kinds",
        nargs="+",
        default=list(DEFAULT_CONTROL_KINDS),
        help=(
            "Control baselines to run. Choices are random_projection, "
            "shuffled_dynamics, and observed_var1."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Base random seed used to derive deterministic control seeds.",
    )


def add_iss_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--em-iters",
        type=int,
        default=50,
        help="Maximum number of EM iterations for ISS fitting.",
    )
    parser.add_argument(
        "--em-tol",
        type=float,
        default=1e-4,
        help="EM convergence tolerance for ISS fitting.",
    )
    parser.add_argument(
        "--em-ridge",
        type=float,
        default=1e-6,
        help="Ridge regularisation used inside ISS EM.",
    )
    parser.add_argument(
        "--allow-time-varying-fallback",
        action="store_true",
        help="Fall back to time-varying filtering if the steady-state solver does not converge.",
    )


def add_prism_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--projection-modes",
        nargs="+",
        default=list(DEFAULT_PRISM_PROJECTION_MODES),
        help="PRISM projection modes to compare. Choices are pca, random, and psi_opt.",
    )
    parser.add_argument(
        "--macro-builder",
        default="hierarchical_complete",
        help="Macrostate builder passed through to the continuous PRISM model.",
    )
    parser.add_argument(
        "--macro-eps",
        type=float,
        default=0.25,
        help="Symmetric KL threshold used when clustering predictive macrostates.",
    )
    parser.add_argument(
        "--macro-bins",
        type=int,
        default=3,
        help="Number of quantile bins per macro dimension for the symboliser.",
    )
    parser.add_argument(
        "--em-iters",
        type=int,
        default=20,
        help="Maximum number of EM iterations for the ISS fit.",
    )
    parser.add_argument(
        "--em-tol",
        type=float,
        default=1e-3,
        help="EM convergence tolerance for the ISS fit.",
    )
    parser.add_argument(
        "--em-ridge",
        type=float,
        default=1e-6,
        help="Small ridge term used inside the ISS fit.",
    )
    parser.add_argument(
        "--iss-mode",
        choices=("steady_state", "time_varying"),
        default="steady_state",
        help="Filtering mode used by the continuous PRISM model.",
    )
    parser.add_argument(
        "--allow-time-varying-fallback",
        action="store_true",
        help="Fall back to time-varying filtering if the steady-state solver does not settle.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Base random seed used to derive deterministic per-trial PRISM seeds.",
    )


def make_trial_metric_evaluator(_config: dict, _inpath: Path):
    def evaluate_trial_fn(X: np.ndarray, _times: np.ndarray) -> list[dict]:
        return [basic_trial_metrics(X)]

    return evaluate_trial_fn


def make_baseline_evaluator(config: dict, _inpath: Path):
    def evaluate_trial_fn(X: np.ndarray, _times: np.ndarray) -> list[dict]:
        return evaluate_trial_baseline(
            X,
            rep_dims=config["rep_dims"],
            train_fraction=config["train_fraction"],
        )

    return evaluate_trial_fn


def make_iss_evaluator(config: dict, _inpath: Path):
    def evaluate_trial_fn(X: np.ndarray, _times: np.ndarray) -> list[dict]:
        return evaluate_trial_iss(
            X,
            rep_dims=config["rep_dims"],
            train_fraction=config["train_fraction"],
            em_iters=config["em_iters"],
            em_tol=config["em_tol"],
            em_ridge=config["em_ridge"],
            allow_time_varying_fallback=config["allow_time_varying_fallback"],
        )

    return evaluate_trial_fn


def make_region_sliding_iss_evaluator(config: dict, inpath: Path):
    has_train_start = config["train_start_ms"] is not None
    has_train_end = config["train_end_ms"] is not None
    if has_train_start != has_train_end:
        raise ValueError("--train-start-ms and --train-end-ms must be set together")
    if not (has_train_start and has_train_end):
        raise ValueError(
            "region-sliding-iss requires explicit --train-start-ms and --train-end-ms"
        )

    subject = extract_subject_id(inpath)
    channel_labels, selected_groups = load_requested_region_groups(subject, config)
    control_groups = build_control_region_groups(
        channel_labels,
        selected_groups,
        n_draws=config["random_control_draws"],
        random_state=subject_seed(subject, base=config["random_state"]),
    )
    all_groups = (*selected_groups, *control_groups)

    def evaluate_trial_fn(X: np.ndarray, times: np.ndarray) -> list[dict]:
        rows: list[dict] = []

        def evaluate_group(group: RegionGroup) -> list[dict]:
            region_rep_dims = rep_dims_for_group(config["rep_dims"], group)
            if not region_rep_dims:
                return []

            return evaluate_trial_iss_baseline_sliding(
                X[:, group.indices],
                times_ms=times,
                rep_dims=region_rep_dims,
                train_start_ms=config["train_start_ms"],
                train_end_ms=config["train_end_ms"],
                target_duration_ms=config["target_duration_ms"],
                step_ms=config["step_ms"],
                target_start_min_ms=config["target_start_min_ms"],
                target_start_max_ms=config["target_start_max_ms"],
                em_iters=config["em_iters"],
                em_tol=config["em_tol"],
                em_ridge=config["em_ridge"],
                allow_time_varying_fallback=config["allow_time_varying_fallback"],
            )

        for group in all_groups:
            rows.extend(
                add_region_metadata(
                    evaluate_group(group),
                    group,
                    include_control_fields=True,
                )
            )

        return rows

    return evaluate_trial_fn


def make_windowed_evaluator(config: dict, _inpath: Path):
    def evaluate_trial_fn(X: np.ndarray, times: np.ndarray) -> list[dict]:
        return evaluate_trial_windows(
            X,
            times_ms=times,
            rep_dims=config["rep_dims"],
            train_fraction=config["train_fraction"],
            windows=config["windows"],
        )

    return evaluate_trial_fn


def make_multiscale_evaluator(config: dict, _inpath: Path):
    def evaluate_trial_fn(X: np.ndarray, times: np.ndarray) -> list[dict]:
        return evaluate_trial_multiscale(
            X,
            times_ms=times,
            rep_dims=config["rep_dims"],
            train_fraction=config["train_fraction"],
            durations_ms=config["durations_ms"],
            center_ms=config["center_ms"],
        )

    return evaluate_trial_fn


def make_centre_sweep_evaluator(config: dict, _inpath: Path):
    sweep_specs = build_centre_sweep_specs(
        centres_ms=config["centres_ms"],
        durations_ms=config["durations_ms"],
    )

    def evaluate_trial_fn(X: np.ndarray, times: np.ndarray) -> list[dict]:
        return evaluate_trial_centre_sweep(
            X,
            times_ms=times,
            rep_dims=config["rep_dims"],
            train_fraction=config["train_fraction"],
            boundary_tolerance_ms=None,
            sweep_specs=sweep_specs,
        )

    return evaluate_trial_fn


def make_context_sweep_evaluator(config: dict, _inpath: Path):
    def evaluate_trial_fn(X: np.ndarray, times: np.ndarray) -> list[dict]:
        return evaluate_trial_context_sweep(
            X,
            times_ms=times,
            rep_dims=config["rep_dims"],
            history_durations_ms=config["history_durations_ms"],
            target_start_ms=config["target_start_ms"],
            target_end_ms=config["target_end_ms"],
            max_train_pairs=config["max_train_pairs"],
        )

    return evaluate_trial_fn


def make_baseline_window_evaluator(config: dict, _inpath: Path):
    def evaluate_trial_fn(X: np.ndarray, times: np.ndarray) -> list[dict]:
        return evaluate_trial_baseline_window(
            X,
            times_ms=times,
            rep_dims=config["rep_dims"],
            train_start_ms=config["train_start_ms"],
            train_end_ms=config["train_end_ms"],
            test_start_ms=config["test_start_ms"],
            test_end_ms=config["test_end_ms"],
            max_train_pairs=config["max_train_pairs"],
        )

    return evaluate_trial_fn


def make_region_sliding_evaluator(config: dict, inpath: Path):
    has_train_start = config["train_start_ms"] is not None
    has_train_end = config["train_end_ms"] is not None
    if has_train_start != has_train_end:
        raise ValueError("--train-start-ms and --train-end-ms must be set together")

    subject = extract_subject_id(inpath)
    channel_labels, selected_groups = load_requested_region_groups(subject, config)
    control_groups = build_control_region_groups(
        channel_labels,
        selected_groups,
        n_draws=config["random_control_draws"],
        random_state=subject_seed(subject, base=config["random_state"]),
    )
    all_groups = (*selected_groups, *control_groups)

    def evaluate_trial_fn(X: np.ndarray, times: np.ndarray) -> list[dict]:
        rows: list[dict] = []
        uses_fixed_baseline = (
            config["train_start_ms"] is not None and config["train_end_ms"] is not None
        )

        def evaluate_group(group: RegionGroup) -> list[dict]:
            region_rep_dims = rep_dims_for_group(config["rep_dims"], group)
            if not region_rep_dims:
                return []

            if uses_fixed_baseline:
                return evaluate_trial_baseline_sliding(
                    X[:, group.indices],
                    times_ms=times,
                    rep_dims=region_rep_dims,
                    train_start_ms=config["train_start_ms"],
                    train_end_ms=config["train_end_ms"],
                    target_duration_ms=config["target_duration_ms"],
                    step_ms=config["step_ms"],
                    target_start_min_ms=config["target_start_min_ms"],
                    target_start_max_ms=config["target_start_max_ms"],
                )

            return evaluate_trial_sliding_context(
                X[:, group.indices],
                times_ms=times,
                rep_dims=region_rep_dims,
                history_ms=config["history_ms"],
                target_duration_ms=config["target_duration_ms"],
                step_ms=config["step_ms"],
                target_start_min_ms=config["target_start_min_ms"],
                target_start_max_ms=config["target_start_max_ms"],
            )

        for group in all_groups:
            rows.extend(
                add_region_metadata(
                    evaluate_group(group),
                    group,
                    include_control_fields=True,
                )
            )

        return rows

    return evaluate_trial_fn


def make_control_evaluator(config: dict, inpath: Path):
    base_seed = subject_seed(inpath.stem, base=config["random_state"])
    trial_seed_rng = np.random.default_rng(base_seed)

    def evaluate_trial_fn(X: np.ndarray, _times: np.ndarray) -> list[dict]:
        trial_seed = int(trial_seed_rng.integers(0, np.iinfo(np.int32).max))
        return evaluate_trial_controls(
            X,
            rep_dims=config["rep_dims"],
            control_kinds=config["control_kinds"],
            train_fraction=config["train_fraction"],
            random_state=trial_seed,
        )

    return evaluate_trial_fn


def make_prism_evaluator(config: dict, inpath: Path):
    base_seed = subject_seed(inpath.stem, base=config["random_state"])
    trial_seed_rng = np.random.default_rng(base_seed)

    def evaluate_trial_fn(X: np.ndarray, _times: np.ndarray) -> list[dict]:
        trial_seed = int(trial_seed_rng.integers(0, np.iinfo(np.int32).max))
        return evaluate_trial_prism(
            X,
            rep_dims=config["rep_dims"],
            projection_modes=config["projection_modes"],
            train_fraction=config["train_fraction"],
            macro_builder=config["macro_builder"],
            macro_eps=config["macro_eps"],
            macro_bins=config["macro_bins"],
            em_iters=config["em_iters"],
            em_tol=config["em_tol"],
            em_ridge=config["em_ridge"],
            iss_mode=config["iss_mode"],
            allow_time_varying_fallback=config["allow_time_varying_fallback"],
            random_state=trial_seed,
        )

    return evaluate_trial_fn


def make_prism_window_evaluator(config: dict, inpath: Path):
    base_seed = subject_seed(inpath.stem, base=config["random_state"])
    trial_seed_rng = np.random.default_rng(base_seed)

    def evaluate_trial_fn(X: np.ndarray, times: np.ndarray) -> list[dict]:
        trial_seed = int(trial_seed_rng.integers(0, np.iinfo(np.int32).max))
        return evaluate_trial_prism_window(
            X,
            times_ms=times,
            rep_dims=config["rep_dims"],
            projection_modes=config["projection_modes"],
            train_start_ms=config["train_start_ms"],
            train_end_ms=config["train_end_ms"],
            test_start_ms=config["test_start_ms"],
            test_end_ms=config["test_end_ms"],
            macro_builder=config["macro_builder"],
            macro_eps=config["macro_eps"],
            macro_bins=config["macro_bins"],
            em_iters=config["em_iters"],
            em_tol=config["em_tol"],
            em_ridge=config["em_ridge"],
            iss_mode=config["iss_mode"],
            allow_time_varying_fallback=config["allow_time_varying_fallback"],
            random_state=trial_seed,
        )

    return evaluate_trial_fn


def make_prism_region_window_evaluator(config: dict, inpath: Path):
    subject = extract_subject_id(inpath)
    _channel_labels, selected_groups = load_requested_region_groups(subject, config)
    base_seed = subject_seed(inpath.stem, base=config["random_state"])
    trial_seed_rng = np.random.default_rng(base_seed)

    def evaluate_trial_fn(X: np.ndarray, times: np.ndarray) -> list[dict]:
        trial_seed = int(trial_seed_rng.integers(0, np.iinfo(np.int32).max))
        rows: list[dict] = []

        for group in selected_groups:
            region_rep_dims = rep_dims_for_group(config["rep_dims"], group)
            if not region_rep_dims:
                continue

            region_seed = subject_seed(group.name, base=trial_seed)
            region_rows = evaluate_trial_prism_window(
                X[:, group.indices],
                times_ms=times,
                rep_dims=region_rep_dims,
                projection_modes=config["projection_modes"],
                train_start_ms=config["train_start_ms"],
                train_end_ms=config["train_end_ms"],
                test_start_ms=config["test_start_ms"],
                test_end_ms=config["test_end_ms"],
                macro_builder=config["macro_builder"],
                macro_eps=config["macro_eps"],
                macro_bins=config["macro_bins"],
                em_iters=config["em_iters"],
                em_tol=config["em_tol"],
                em_ridge=config["em_ridge"],
                iss_mode=config["iss_mode"],
                allow_time_varying_fallback=config["allow_time_varying_fallback"],
                random_state=region_seed,
            )
            rows.extend(add_region_metadata(region_rows, group))

        return rows

    return evaluate_trial_fn


def describe_trial_metrics(df: pd.DataFrame, _subject: str) -> str:
    return f"Processed {len(df)} trials"


def describe_baseline(df: pd.DataFrame, subject: str) -> str:
    return f"Processed {df['trial_idx'].nunique()} trials for {subject}"


def describe_iss(df: pd.DataFrame, subject: str) -> str:
    return f"Processed {df['trial_idx'].nunique()} trials (ISS) for {subject}"


def describe_region_sliding_iss(df: pd.DataFrame, subject: str) -> str:
    n_windows = df["target_start_ms"].nunique()
    n_regions = df["region_name"].nunique()
    return (
        f"Processed {df['trial_idx'].nunique()} trials "
        f"x {n_regions} channel groups x {n_windows} sliding windows (ISS) for {subject}"
    )


def describe_windowed(df: pd.DataFrame, subject: str) -> str:
    return (
        f"Processed {df['trial_idx'].nunique()} trials "
        f"x {df['window_name'].nunique()} windows for {subject}"
    )


def describe_multiscale(df: pd.DataFrame, subject: str) -> str:
    return (
        f"Processed {df['trial_idx'].nunique()} trials "
        f"x {df['scale_name'].nunique()} scales for {subject}"
    )


def describe_centre_sweep(df: pd.DataFrame, subject: str) -> str:
    n_windows = df[["duration_ms", "center_ms"]].drop_duplicates().shape[0]
    return (
        f"Processed {df['trial_idx'].nunique()} trials "
        f"x {n_windows} windows for {subject}"
    )


def describe_context_sweep(df: pd.DataFrame, subject: str) -> str:
    n_histories = df["history_ms"].nunique()
    return (
        f"Processed {df['trial_idx'].nunique()} trials "
        f"x {n_histories} context settings for {subject}"
    )


def describe_region_sliding(df: pd.DataFrame, subject: str) -> str:
    n_windows = df["target_start_ms"].nunique()
    n_regions = df["region_name"].nunique()
    return (
        f"Processed {df['trial_idx'].nunique()} trials "
        f"x {n_regions} channel groups x {n_windows} sliding windows for {subject}"
    )


def describe_control(df: pd.DataFrame, subject: str) -> str:
    controls = ", ".join(sorted(df["control_kind"].dropna().astype(str).unique()))
    return f"Processed {df['trial_idx'].nunique()} trials for {subject} across {controls}"


def describe_prism(df: pd.DataFrame, subject: str) -> str:
    families = ", ".join(sorted(df["model_family"].dropna().astype(str).unique()))
    return f"Processed {df['trial_idx'].nunique()} trials for {subject} across {families}"


EXPERIMENTS = (
    ExperimentSpec(
        name="trial-metrics",
        description="Run the older descriptive trial metrics for one or all subjects.",
        default_outdir="./data/results",
        combined_filename="all_subjects_trial_metrics.csv",
        output_suffix="_trial_metrics.csv",
        output_columns=TRIAL_METRIC_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_trial_metric_evaluator,
        sort_columns=("subject", "trial_idx"),
        load_subject_fn=load_and_prepare_subject,
        time_bounds_keys=("tmin_ms", "tmax_ms"),
        describe_run=describe_trial_metrics,
    ),
    ExperimentSpec(
        name="baseline",
        description="Run the PCA + VAR(1) trial baseline for one or all subjects.",
        default_outdir="./data/results_baseline",
        combined_filename="all_subjects_trial_baseline.csv",
        output_suffix="_trial_baseline.csv",
        output_columns=COMMON_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_baseline_evaluator,
        sort_columns=("subject", "trial_idx", "rep_dim"),
        time_bounds_keys=("tmin_ms", "tmax_ms"),
        train_fraction_help="Fraction of each trial used for training along the time axis.",
        rep_dims_default=DEFAULT_REP_DIMS,
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "train_fraction": args.train_fraction,
        },
        describe_run=describe_baseline,
    ),
    ExperimentSpec(
        name="iss",
        description="Run the PCA + steady-state ISS (EM) trial evaluator for one or all subjects.",
        default_outdir="./data/results_iss",
        combined_filename="all_subjects_trial_iss.csv",
        output_suffix="_trial_iss.csv",
        output_columns=COMMON_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_iss_evaluator,
        sort_columns=("subject", "trial_idx", "rep_dim"),
        time_bounds_keys=("tmin_ms", "tmax_ms"),
        train_fraction_help="Fraction of each trial used for training along the time axis.",
        rep_dims_default=DEFAULT_REP_DIMS,
        add_arguments=add_iss_arguments,
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "train_fraction": args.train_fraction,
            "em_iters": args.em_iters,
            "em_tol": args.em_tol,
            "em_ridge": args.em_ridge,
            "allow_time_varying_fallback": args.allow_time_varying_fallback,
        },
        describe_run=describe_iss,
    ),
    ExperimentSpec(
        name="windowed",
        description="Run the stimulus-locked windowed baseline for one or all subjects.",
        default_outdir="./data/results_baseline/windowed",
        combined_filename="all_subjects_windowed_baseline.csv",
        output_suffix="_windowed_baseline.csv",
        output_columns=WINDOW_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_windowed_evaluator,
        sort_columns=("subject", "trial_idx", "window_name", "rep_dim"),
        train_fraction_help="Fraction of each window used for training along the time axis.",
        rep_dims_default=DEFAULT_REP_DIMS,
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "train_fraction": args.train_fraction,
            "windows": DEFAULT_STIM_LOCKED_WINDOWS,
        },
        describe_run=describe_windowed,
    ),
    ExperimentSpec(
        name="multiscale",
        description="Run the centred duration-sweep baseline for one or all subjects.",
        default_outdir="./data/results_baseline/multiscale",
        combined_filename="all_subjects_multiscale_baseline.csv",
        output_suffix="_multiscale_baseline.csv",
        output_columns=SCALE_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_multiscale_evaluator,
        sort_columns=("subject", "trial_idx", "duration_ms", "rep_dim"),
        train_fraction_help="Fraction of each scale window used for training along the time axis.",
        rep_dims_default=DEFAULT_REP_DIMS,
        add_arguments=add_multiscale_arguments,
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "train_fraction": args.train_fraction,
            "durations_ms": tuple(args.durations_ms),
            "center_ms": args.centre_ms,
        },
        describe_run=describe_multiscale,
    ),
    ExperimentSpec(
        name="centre-sweep",
        description="Run the duration x centre sweep baseline for one or all subjects.",
        default_outdir="./data/results_baseline/center_sweep",
        combined_filename="all_subjects_center_sweep_baseline.csv",
        output_suffix="_center_sweep_baseline.csv",
        output_columns=SCALE_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_centre_sweep_evaluator,
        sort_columns=("subject", "trial_idx", "duration_ms", "center_ms", "rep_dim"),
        progress_every=50,
        train_fraction_help="Fraction of each window used for training along the time axis.",
        rep_dims_default=DEFAULT_REP_DIMS,
        add_arguments=add_centre_sweep_arguments,
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "train_fraction": args.train_fraction,
            "durations_ms": tuple(args.durations_ms),
            "centres_ms": tuple(args.centres_ms),
        },
        describe_run=describe_centre_sweep,
        aliases=("center-sweep",),
    ),
    ExperimentSpec(
        name="context-sweep",
        description="Run the fixed-target context sweep baseline for one or all subjects.",
        default_outdir="./data/results_baseline/context_sweep",
        combined_filename="all_subjects_context_sweep_baseline.csv",
        output_suffix="_context_sweep_baseline.csv",
        output_columns=CONTEXT_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_context_sweep_evaluator,
        sort_columns=("subject", "trial_idx", "history_ms", "rep_dim"),
        add_arguments=add_context_sweep_arguments,
        rep_dims_default=DEFAULT_REP_DIMS,
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "history_durations_ms": tuple(args.history_durations_ms),
            "target_start_ms": args.target_start_ms,
            "target_end_ms": args.target_end_ms,
            "max_train_pairs": args.max_train_pairs,
        },
        describe_run=describe_context_sweep,
    ),
    ExperimentSpec(
        name="baseline-window",
        description="Run the baseline on one explicit train/test window pair for one or all subjects.",
        default_outdir="./data/results_baseline/window_compare",
        combined_filename="all_subjects_trial_baseline.csv",
        output_suffix="_trial_baseline.csv",
        output_columns=TRAIN_TEST_WINDOW_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_baseline_window_evaluator,
        sort_columns=("subject", "trial_idx", "test_start_ms", "rep_dim"),
        add_arguments=add_train_test_window_arguments,
        rep_dims_default=DEFAULT_REP_DIMS,
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "train_start_ms": args.train_start_ms,
            "train_end_ms": args.train_end_ms,
            "test_start_ms": args.test_start_ms,
            "test_end_ms": args.test_end_ms,
            "max_train_pairs": args.max_train_pairs,
        },
        describe_run=describe_baseline,
    ),
    ExperimentSpec(
        name="region-sliding",
        description="Run the fixed-size sliding baseline across all channels and scalp regions.",
        default_outdir="./data/results_baseline/region_sliding",
        combined_filename="all_subjects_region_sliding_baseline.csv",
        output_suffix="_region_sliding_baseline.csv",
        output_columns=REGION_SLIDING_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_region_sliding_evaluator,
        sort_columns=("subject", "trial_idx", "region_name", "target_start_ms", "rep_dim"),
        add_arguments=add_region_sliding_arguments,
        rep_dims_default=(2, 4),
        rep_dims_help=(
            "Representation dimensions to evaluate. Defaults are kept small so every scalp "
            "region has enough channels for PCA."
        ),
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "history_ms": args.history_ms,
            "target_duration_ms": args.target_duration_ms,
            "step_ms": args.step_ms,
            "target_start_min_ms": args.target_start_min_ms,
            "target_start_max_ms": args.target_start_max_ms,
            "train_start_ms": args.train_start_ms,
            "train_end_ms": args.train_end_ms,
            "regions": tuple(args.regions),
            "derivatives_dir": Path(args.derivatives_dir),
            "random_control_draws": args.random_control_draws,
            "random_state": args.random_state,
        },
        describe_run=describe_region_sliding,
    ),
    ExperimentSpec(
        name="region-sliding-iss",
        description=(
            "Run the steady-state ISS sliding analysis across scalp regions "
            "with a fixed pre-stimulus training window."
        ),
        default_outdir="./data/results_iss/region_sliding_iss",
        combined_filename="all_subjects_region_sliding_iss.csv",
        output_suffix="_region_sliding_iss.csv",
        output_columns=REGION_SLIDING_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_region_sliding_iss_evaluator,
        sort_columns=("subject", "trial_idx", "region_name", "target_start_ms", "rep_dim"),
        add_arguments=lambda p: (add_region_sliding_arguments(p), add_iss_arguments(p)),
        rep_dims_default=(2, 4),
        rep_dims_help=(
            "Representation dimensions to evaluate. Defaults are kept small so every "
            "scalp region has enough channels for PCA and ISS EM fitting."
        ),
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "history_ms": args.history_ms,
            "target_duration_ms": args.target_duration_ms,
            "step_ms": args.step_ms,
            "target_start_min_ms": args.target_start_min_ms,
            "target_start_max_ms": args.target_start_max_ms,
            "train_start_ms": args.train_start_ms,
            "train_end_ms": args.train_end_ms,
            "regions": tuple(args.regions),
            "derivatives_dir": Path(args.derivatives_dir),
            "random_control_draws": args.random_control_draws,
            "random_state": args.random_state,
            "em_iters": args.em_iters,
            "em_tol": args.em_tol,
            "em_ridge": args.em_ridge,
            "allow_time_varying_fallback": args.allow_time_varying_fallback,
        },
        describe_run=describe_region_sliding_iss,
    ),
    ExperimentSpec(
        name="control",
        description="Run matched control baselines for one or all subjects.",
        default_outdir="./data/results_baseline/controls",
        combined_filename="all_subjects_control_baseline.csv",
        output_suffix="_control_baseline.csv",
        output_columns=CONTROL_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_control_evaluator,
        sort_columns=("subject", "trial_idx", "control_kind", "rep_dim"),
        time_bounds_keys=("tmin_ms", "tmax_ms"),
        train_fraction_help="Fraction of each trial used for training along the time axis.",
        rep_dims_default=DEFAULT_REP_DIMS,
        add_arguments=add_control_arguments,
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "control_kinds": tuple(args.control_kinds),
            "train_fraction": args.train_fraction,
            "random_state": args.random_state,
        },
        describe_run=describe_control,
    ),
    ExperimentSpec(
        name="prism",
        description="Run the continuous PRISM comparison against the trial baseline.",
        default_outdir="./data/results_prism",
        combined_filename="all_subjects_trial_prism.csv",
        output_suffix="_trial_prism.csv",
        output_columns=PRISM_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_prism_evaluator,
        sort_columns=("subject", "trial_idx", "model_family", "rep_dim"),
        time_bounds_keys=("tmin_ms", "tmax_ms"),
        train_fraction_help="Fraction of each trial used for training along the time axis.",
        rep_dims_default=DEFAULT_REP_DIMS,
        add_arguments=add_prism_arguments,
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "projection_modes": tuple(args.projection_modes),
            "train_fraction": args.train_fraction,
            "macro_builder": args.macro_builder,
            "macro_eps": args.macro_eps,
            "macro_bins": args.macro_bins,
            "em_iters": args.em_iters,
            "em_tol": args.em_tol,
            "em_ridge": args.em_ridge,
            "iss_mode": args.iss_mode,
            "allow_time_varying_fallback": args.allow_time_varying_fallback,
            "random_state": args.random_state,
        },
        describe_run=describe_prism,
    ),
    ExperimentSpec(
        name="prism-window",
        description="Run the continuous PRISM comparison on one explicit train/test window pair.",
        default_outdir="./data/results_prism/window_compare",
        combined_filename="all_subjects_trial_prism.csv",
        output_suffix="_trial_prism.csv",
        output_columns=PRISM_WINDOW_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_prism_window_evaluator,
        sort_columns=("subject", "trial_idx", "model_family", "test_start_ms", "rep_dim"),
        time_bounds_keys=("trial_tmin_ms", "trial_tmax_ms"),
        rep_dims_default=DEFAULT_REP_DIMS,
        add_arguments=add_prism_window_arguments,
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "projection_modes": tuple(args.projection_modes),
            "train_start_ms": args.train_start_ms,
            "train_end_ms": args.train_end_ms,
            "test_start_ms": args.test_start_ms,
            "test_end_ms": args.test_end_ms,
            "macro_builder": args.macro_builder,
            "macro_eps": args.macro_eps,
            "macro_bins": args.macro_bins,
            "em_iters": args.em_iters,
            "em_tol": args.em_tol,
            "em_ridge": args.em_ridge,
            "iss_mode": args.iss_mode,
            "allow_time_varying_fallback": args.allow_time_varying_fallback,
            "random_state": args.random_state,
        },
        describe_run=describe_prism,
    ),
    ExperimentSpec(
        name="prism-region-window",
        description="Run the continuous PRISM comparison on explicit train/test windows within scalp regions.",
        default_outdir="./data/results_prism/region_window",
        combined_filename="all_subjects_region_window_prism.csv",
        output_suffix="_region_window_prism.csv",
        output_columns=PRISM_REGION_WINDOW_OUTPUT_COLUMNS,
        evaluate_trial_factory=make_prism_region_window_evaluator,
        sort_columns=("subject", "trial_idx", "region_name", "model_family", "rep_dim"),
        time_bounds_keys=("trial_tmin_ms", "trial_tmax_ms"),
        rep_dims_default=(2, 4),
        rep_dims_help=(
            "Representation dimensions to evaluate. Defaults are kept small so every scalp "
            "region has enough channels for the latent fit."
        ),
        add_arguments=add_prism_region_window_arguments,
        config_from_args=lambda args: {
            "rep_dims": tuple(args.rep_dims),
            "projection_modes": tuple(args.projection_modes),
            "train_start_ms": args.train_start_ms,
            "train_end_ms": args.train_end_ms,
            "test_start_ms": args.test_start_ms,
            "test_end_ms": args.test_end_ms,
            "regions": tuple(args.regions),
            "derivatives_dir": Path(args.derivatives_dir),
            "macro_builder": args.macro_builder,
            "macro_eps": args.macro_eps,
            "macro_bins": args.macro_bins,
            "em_iters": args.em_iters,
            "em_tol": args.em_tol,
            "em_ridge": args.em_ridge,
            "iss_mode": args.iss_mode,
            "allow_time_varying_fallback": args.allow_time_varying_fallback,
            "random_state": args.random_state,
        },
        describe_run=describe_prism,
    ),
)


EXPERIMENT_INDEX = {
    name: spec
    for spec in EXPERIMENTS
    for name in (spec.name, *spec.aliases)
}


def get_experiment(name: str) -> ExperimentSpec:
    return EXPERIMENT_INDEX[name]
