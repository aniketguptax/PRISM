"""Experiment runner for discrete and continuous PRISM pipelines."""

import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from prism.metrics import log_loss, n_states, statistical_complexity, unifilarity_score
from prism.metrics.branching import mean_branching_entropy_weighted
from prism.metrics.gaussian_predictive import gaussian_log_loss
from prism.metrics.graph import DotStyle, dot_to_png, save_dot, to_dot, to_edge_list
from prism.processes.protocols import Process
from prism.reconstruction.kalman_iss import GaussianPredictiveStateModel
from prism.reconstruction.protocols import PredictiveStateModel, Reconstructor
from prism.representations.protocols import Representation
from prism.types import Obs
from prism.utils.io import save_csv, save_json


def _extract_k(rep_name: str) -> Optional[int]:
    match = re.search(r"(?:last_|iss_d)(\d+)", rep_name)
    if match:
        return int(match.group(1))
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
) -> Dict[str, float]:
    logloss = log_loss(x_test, rep, model)
    return {
        "logloss": logloss,
        "gaussian_logloss": math.nan,
        "n_states": float(n_states(model)),
        "C_mu_empirical": statistical_complexity(model),
        "unifilarity_score": unifilarity_score(model),
        "branch_entropy": mean_branching_entropy_weighted(model, log_base=2.0),
    }


def _continuous_metrics(
    x_train: Sequence[Obs],
    x_test: Sequence[Obs],
    model: GaussianPredictiveStateModel,
) -> Dict[str, float]:
    heldout_nll = gaussian_log_loss(x_test, model, context=x_train)
    return {
        "logloss": heldout_nll,
        "gaussian_logloss": heldout_nll,
        "n_states": float(n_states(model)),
        "C_mu_empirical": math.nan,
        "unifilarity_score": math.nan,
        "branch_entropy": math.nan,
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
        "reconstructor",
        "logloss",
        "gaussian_logloss",
        "n_states",
        "C_mu_empirical",
        "unifilarity_score",
        "branch_entropy",
    ]

    cond = _cond_fields(condition, process.name)
    condition_id = cond["condition_id"]
    transitions_dir = outdir / "transitions"

    for seed in seeds:
        sample = process.sample(length=length, seed=seed)
        x = sample.x
        if len(x) < 2:
            raise ValueError(
                f"Process {process.name} produced {len(x)} samples for length={length}; need at least 2."
            )
        split = int(len(x) * train_frac)
        split = max(1, min(split, len(x) - 1))
        x_train, x_test = x[:split], x[split:]

        rows: list[dict[str, Any]] = []
        for rep in representations:
            model = reconstructor.fit(x_train, rep, seed=seed)
            metrics: Dict[str, float]
            if isinstance(model, GaussianPredictiveStateModel):
                metrics = _continuous_metrics(x_train, x_test, model)
                if save_transitions and (transitions_rep_name is None or rep.name == transitions_rep_name):
                    raise ValueError(
                        "Transition export is only supported for discrete state models. "
                        "Continuous Kalman ISS runs do not define discrete symbol-conditioned transitions."
                    )
            else:
                if not isinstance(model, PredictiveStateModel):
                    raise TypeError(
                        f"Unsupported model type returned by {reconstructor.name}: {type(model).__name__}."
                    )
                metrics = _discrete_metrics(x_train, x_test, rep, model)
                if save_transitions and (transitions_rep_name is None or rep.name == transitions_rep_name):
                    edges = to_edge_list(model)
                    transitions_dir.mkdir(exist_ok=True)
                    json_path = transitions_dir / f"{condition_id}__transitions_{rep.name}_seed{seed}.json"
                    dot_path = transitions_dir / f"{condition_id}__transitions_{rep.name}_seed{seed}.dot"
                    png_path = transitions_dir / f"{condition_id}__transitions_{rep.name}_seed{seed}.png"
                    save_json(json_path, edges)
                    dot = to_dot(
                        edges,
                        graph_name=f"{process.name}_{rep.name}_seed{seed}",
                        label=f"{process.name} | {rep.name} | seed={seed} | {condition_id}",
                        style=DotStyle(rankdir="LR"),
                    )
                    save_dot(dot_path, dot)
                    dot_to_png(dot_path, png_path)

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
                    "reconstructor": reconstructor.name,
                    **metrics,
                }
            )
        save_csv(metrics_path, rows, append=True, fieldnames=fieldnames)
