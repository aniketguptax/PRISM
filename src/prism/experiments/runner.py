import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from prism.metrics import log_loss_with_context, n_states, statistical_complexity, unifilarity_score
from prism.metrics.branching import mean_branching_entropy_weighted
from prism.metrics.gaussian_predictive import gaussian_log_loss, macro_predictive_log_loss
from prism.metrics.graph import DotStyle, save_dot, to_dot, to_edge_list
from prism.viz import render_from_edges
from prism.processes.protocols import Process
from prism.reconstruction.kalman_iss import GaussianPredictiveStateModel
from prism.reconstruction.protocols import PredictiveStateModel, Reconstructor
from prism.representations.protocols import Representation
from prism.types import Obs
from prism.utils.io import save_csv, save_json

LOGGER = logging.getLogger(__name__)


def _extract_k(rep_name: str) -> Optional[int]:
    match = re.search(r"(?:last_|iss_d)(\d+)", rep_name)
    if match:
        return int(match.group(1))
    return None


def _extract_dv(rep_name: str) -> Optional[int]:
    match = re.search(r"_dv(\d+)", rep_name)
    if match:
        return int(match.group(1))
    if rep_name.startswith("iss_d"):
        return 1
    return None


def _cond_fields(condition: Optional[Dict[str, Any]], process_name_fallback: str) -> Dict[str, Any]:
    cond = condition or {}
    base_process = str(cond.get("base_process", process_name_fallback))
    flip_p = float(cond.get("flip_p", 0.0))
    subsample_step = int(cond.get("subsample_step", 1))
    wrappers = str(cond.get("wrappers", ""))
    condition_id = str(cond.get("condition_id", f"flip{flip_p:g}_sub{subsample_step}"))
    return {
        "base_process": base_process,
        "flip_p": flip_p,
        "subsample_step": subsample_step,
        "wrappers": wrappers,
        "condition_id": condition_id,
    }


def _discrete_metrics(
    x_train: Sequence[Obs],
    x_test: Sequence[Obs],
    rep: Representation,
    model: PredictiveStateModel,
) -> Dict[str, Any]:
    if not model.valid:
        reason = model.invalid_reason or "invalid predictive model"
        LOGGER.warning("Discrete model invalid for representation %s: %s", rep.name, reason)
        return {
            "logloss": math.nan,
            "gaussian_logloss": math.nan,
            "macro_logloss": math.nan,
            "n_states": math.nan,
            "C_mu_empirical": math.nan,
            "unifilarity_score": math.nan,
            "branch_entropy": math.nan,
            "psi_opt": math.nan,
            "psi_macro_dim": math.nan,
            "psi_optimiser": "",
            "macro_dim": math.nan,
            "obs_dim": math.nan,
            "projection_mode": "",
            "macro_builder": "",
            "macro_eps": math.nan,
            "macro_bins": math.nan,
            "macro_symboliser": "",
            "model_selection": False,
            "selection_score": "",
            "selection_value": math.nan,
            "selection_predictive": math.nan,
            "selection_stability": math.nan,
            "selection_n_states_var": math.nan,
            "selection_unifilarity_var": math.nan,
            "selection_branch_entropy_var": math.nan,
            "selection_n_states_mean": math.nan,
            "selection_unifilarity_mean": math.nan,
            "selection_branch_entropy_mean": math.nan,
            "selection_macro_time_mean_s": math.nan,
            "selection_macro_time_max_s": math.nan,
            "selection_distance_matrix_mb_est": math.nan,
            "selection_n_pred": math.nan,
            "latent_dim": math.nan,
            "iss_mode": "",
            "macro_build_time_s": math.nan,
            "macro_distance_matrix_mb_est": math.nan,
            "macro_n_pred": math.nan,
        }

    # Fit is train-only. Held-out evaluation uses train context solely to build
    # boundary-crossing representational states z_t = phi_k(x_{1:t}) on test.
    try:
        logloss = log_loss_with_context(x_train, x_test, rep, model)
    except ValueError as exc:
        raise ValueError(
            f"Held-out discrete evaluation failed for representation {rep.name}: {exc}"
        ) from exc
    complexity = statistical_complexity(model) if model.pi else math.nan
    if not model.pi:
        LOGGER.warning("Occupancy is undefined for representation %s; setting C_mu to NaN.", rep.name)
    if model.transitions and sum(model.sa_counts.values()) > 0:
        unif = unifilarity_score(model)
        branch = mean_branching_entropy_weighted(model, log_base=2.0)
    else:
        LOGGER.warning("Transitions are undefined for representation %s; setting branching metrics to NaN.", rep.name)
        unif = math.nan
        branch = math.nan

    return {
        "logloss": logloss,
        "gaussian_logloss": math.nan,
        "macro_logloss": math.nan,
        "n_states": float(n_states(model)),
        "C_mu_empirical": complexity,
        "unifilarity_score": unif,
        "branch_entropy": branch,
        "psi_opt": math.nan,
        "psi_macro_dim": math.nan,
        "psi_optimiser": "",
        "macro_dim": math.nan,
        "obs_dim": math.nan,
        "projection_mode": "",
        "macro_builder": "",
        "macro_eps": math.nan,
        "macro_bins": math.nan,
        "macro_symboliser": "",
        "model_selection": False,
        "selection_score": "",
        "selection_value": math.nan,
        "selection_predictive": math.nan,
        "selection_stability": math.nan,
        "selection_n_states_var": math.nan,
        "selection_unifilarity_var": math.nan,
        "selection_branch_entropy_var": math.nan,
        "selection_n_states_mean": math.nan,
        "selection_unifilarity_mean": math.nan,
        "selection_branch_entropy_mean": math.nan,
        "selection_macro_time_mean_s": math.nan,
        "selection_macro_time_max_s": math.nan,
        "selection_distance_matrix_mb_est": math.nan,
        "selection_n_pred": math.nan,
        "latent_dim": math.nan,
        "iss_mode": "",
        "macro_build_time_s": math.nan,
        "macro_distance_matrix_mb_est": math.nan,
        "macro_n_pred": math.nan,
    }


def _continuous_metrics(
    x_train: Sequence[Obs],
    x_test: Sequence[Obs],
    model: GaussianPredictiveStateModel,
) -> Dict[str, Any]:
    heldout_nll = gaussian_log_loss(x_test, model, context=x_train)
    macro_nll = macro_predictive_log_loss(x_test, model, context=x_train)
    if math.isnan(heldout_nll):
        LOGGER.warning("Gaussian log-loss undefined for continuous run; setting to NaN.")
    if math.isnan(macro_nll):
        LOGGER.warning("Macro predictive log-loss undefined for continuous run; setting to NaN.")

    complexity = statistical_complexity(model) if model.pi else math.nan
    if not model.pi:
        LOGGER.warning("Continuous occupancy is empty; setting C_mu to NaN.")

    if model.transitions and sum(model.sa_counts.values()) > 0:
        unif = unifilarity_score(model)
        branch = mean_branching_entropy_weighted(model, log_base=2.0)
    else:
        LOGGER.warning("Continuous transitions are empty; setting branching metrics to NaN.")
        unif = math.nan
        branch = math.nan

    return {
        "logloss": heldout_nll,
        "gaussian_logloss": heldout_nll,
        "macro_logloss": macro_nll,
        "n_states": float(n_states(model)) if model.n_macro_states > 0 else math.nan,
        "C_mu_empirical": complexity,
        "unifilarity_score": unif,
        "branch_entropy": branch,
        "psi_opt": float(model.psi_opt),
        "psi_macro_dim": float(model.psi_macro_dim),
        "psi_optimiser": model.psi_optimiser,
        "macro_dim": model.macro_dim,
        "obs_dim": model.obs_dim,
        "projection_mode": model.projection_mode,
        "macro_builder": model.macro_builder,
        "macro_eps": model.macro_eps,
        "macro_bins": model.macro_bins,
        "macro_symboliser": model.macro_symboliser,
        "model_selection": model.model_selection,
        "selection_score": model.selection_score,
        "selection_value": model.selection_value,
        "selection_predictive": model.selection_predictive,
        "selection_stability": model.selection_stability,
        "selection_n_states_var": model.selection_n_states_var,
        "selection_unifilarity_var": model.selection_unifilarity_var,
        "selection_branch_entropy_var": model.selection_branch_entropy_var,
        "selection_n_states_mean": model.selection_n_states_mean,
        "selection_unifilarity_mean": model.selection_unifilarity_mean,
        "selection_branch_entropy_mean": model.selection_branch_entropy_mean,
        "selection_macro_time_mean_s": model.selection_macro_time_mean_s,
        "selection_macro_time_max_s": model.selection_macro_time_max_s,
        "selection_distance_matrix_mb_est": model.selection_distance_matrix_mb_est,
        "selection_n_pred": model.selection_n_pred,
        "latent_dim": model.latent_dim,
        "iss_mode": model.iss_mode,
        "macro_build_time_s": model.macro_build_time_s,
        "macro_distance_matrix_mb_est": model.macro_distance_matrix_mb_est,
        "macro_n_pred": model.macro_n_pred,
    }


def run_experiment(
    process: Process,
    reconstructor: Reconstructor[object],
    representations: Sequence[Representation],
    length: int,
    train_frac: float,
    seeds: Sequence[int],
    outdir: Path,
    condition: Optional[Dict[str, Any]] = None,
    save_transitions: bool = False,
    transitions_rep_name: Optional[str] = None,
) -> None:
    """Run a full process/reconstructor/representation sweep and persist metrics."""
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "runs.csv"
    fieldnames = [
        "seed",
        "length",
        "train_frac",
        "process",
        "base_process",
        "eps",
        "flip_p",
        "subsample_step",
        "wrappers",
        "condition_id",
        "representation",
        "k",
        "dv",
        "reconstructor",
        "logloss",
        "gaussian_logloss",
        "macro_logloss",
        "n_states",
        "C_mu_empirical",
        "unifilarity_score",
        "branch_entropy",
        "psi_opt",
        "psi_macro_dim",
        "psi_optimiser",
        "macro_dim",
        "obs_dim",
        "projection_mode",
        "macro_builder",
        "macro_eps",
        "macro_bins",
        "macro_symboliser",
        "model_selection",
        "selection_score",
        "selection_value",
        "selection_predictive",
        "selection_stability",
        "selection_n_states_var",
        "selection_unifilarity_var",
        "selection_branch_entropy_var",
        "selection_n_states_mean",
        "selection_unifilarity_mean",
        "selection_branch_entropy_mean",
        "selection_macro_time_mean_s",
        "selection_macro_time_max_s",
        "selection_distance_matrix_mb_est",
        "selection_n_pred",
        "latent_dim",
        "iss_mode",
        "macro_build_time_s",
        "macro_distance_matrix_mb_est",
        "macro_n_pred",
        "model_valid",
        "invalid_reason",
    ]

    cond = _cond_fields(condition, process.name)
    condition_id = cond["condition_id"]
    transitions_dir = outdir / "transitions"
    n_seeds = len(seeds)
    n_reps = len(representations)
    total_tasks = n_seeds * n_reps
    completed_tasks = 0
    run_start = time.perf_counter()

    LOGGER.info(
        "Experiment start | process=%s reconstructor=%s condition=%s seeds=%d representations=%d",
        process.name,
        reconstructor.name,
        condition_id,
        n_seeds,
        n_reps,
    )

    def _fmt_value(value: Any, *, digits: int = 4) -> str:
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return "nan"
            return f"{number:.{digits}f}"
        return str(value)

    for seed_idx, seed in enumerate(seeds, start=1):
        sample = process.sample(length=length, seed=seed)
        x = sample.x
        if len(x) < 2:
            raise ValueError(
                f"Process {process.name} produced {len(x)} samples for length={length}; need at least 2."
            )
        split = int(len(x) * train_frac)
        split = max(1, min(split, len(x) - 1))
        x_train, x_test = x[:split], x[split:]
        LOGGER.info(
            "Seed %d/%d | seed=%d train=%d test=%d",
            seed_idx,
            n_seeds,
            seed,
            len(x_train),
            len(x_test),
        )

        rows: list[dict[str, Any]] = []
        for rep_idx, rep in enumerate(representations, start=1):
            rep_start = time.perf_counter()
            model = reconstructor.fit(x_train, rep, seed=seed)
            metrics: Dict[str, Any]
            model_valid = True
            invalid_reason = ""
            if isinstance(model, GaussianPredictiveStateModel):
                metrics = _continuous_metrics(x_train, x_test, model)
            else:
                if not isinstance(model, PredictiveStateModel):
                    raise TypeError(
                        f"Unsupported model type returned by {reconstructor.name}: {type(model).__name__}."
                    )
                metrics = _discrete_metrics(x_train, x_test, rep, model)
                model_valid = model.valid
                invalid_reason = model.invalid_reason
            if save_transitions and (transitions_rep_name is None or rep.name == transitions_rep_name):
                edges = to_edge_list(model)
                transitions_dir.mkdir(exist_ok=True)
                stem = f"{condition_id}__transitions_{rep.name}_seed{seed}"
                json_path = transitions_dir / f"{stem}.json"
                meta_path = transitions_dir / f"{stem}_meta.json"
                dot_path = transitions_dir / f"{stem}.dot"
                save_json(json_path, edges)
                emission = {
                    int(k): float(v) for k, v in getattr(model, "p_next_one", {}).items()
                }
                occupancy = {int(k): float(v) for k, v in getattr(model, "pi", {}).items()}
                save_json(meta_path, {"emission": emission, "occupancy": occupancy})
                dot = to_dot(
                    edges,
                    graph_name=f"{process.name}_{rep.name}_seed{seed}",
                    label=None,
                    style=DotStyle(rankdir="LR"),
                )
                save_dot(dot_path, dot)
                render_from_edges(
                    edges,
                    transitions_dir / stem,
                    emission=emission,
                    occupancy=occupancy,
                )

            rows.append(
                {
                    "seed": seed,
                    "length": length,
                    "train_frac": train_frac,
                    "process": process.name,
                    "eps": getattr(reconstructor, "eps", None),
                    **cond,
                    "representation": rep.name,
                    "k": _extract_k(rep.name),
                    "dv": _extract_dv(rep.name),
                    "reconstructor": reconstructor.name,
                    "model_valid": model_valid,
                    "invalid_reason": invalid_reason,
                    **metrics,
                }
            )
            completed_tasks += 1
            elapsed_rep = time.perf_counter() - rep_start
            LOGGER.info(
                "Progress %d/%d | seed=%d rep=%d/%d (%s) logloss=%s n_states=%s elapsed=%.2fs",
                completed_tasks,
                total_tasks,
                seed,
                rep_idx,
                n_reps,
                rep.name,
                _fmt_value(metrics.get("logloss", math.nan)),
                _fmt_value(metrics.get("n_states", math.nan), digits=2),
                elapsed_rep,
            )
        save_csv(metrics_path, rows, append=True, fieldnames=fieldnames)
        LOGGER.info("Seed %d/%d complete | wrote %d rows to %s", seed_idx, n_seeds, len(rows), metrics_path)

    LOGGER.info(
        "Experiment complete | condition=%s total_rows=%d elapsed=%.2fs output=%s",
        condition_id,
        total_tasks,
        time.perf_counter() - run_start,
        metrics_path,
    )
