from __future__ import annotations

import argparse
import csv
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Sequence

from prism.analysis.summarise import summarise
from prism.experiments.runner import run_experiment
from prism.processes.continuous_file import ContinuousFile, load_timeseries
from prism.reconstruction.kalman_iss import KalmanISSReconstructor
from prism.representations.continuous import ISSDim
from prism.utils.io import save_json
from prism.utils.logging import configure_logging


def _default_d_grid(train_len: int, obs_dim: int, max_d: int) -> list[int]:
    cap = max(1, min(max_d, obs_dim, max(1, train_len // 5)))
    if cap <= 4:
        return list(range(1, cap + 1))
    base = [1, 2, 3, 4, cap]
    return sorted({d for d in base if 1 <= d <= cap})


def _default_dv_grid(obs_dim: int, max_dv: int) -> list[int]:
    cap = max(1, min(obs_dim, max_dv))
    return list(range(1, cap + 1))


def _parse_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(text: str | None) -> float:
    if text is None:
        return math.nan
    try:
        val = float(text)
    except (TypeError, ValueError):
        return math.nan
    return val if math.isfinite(val) else math.nan


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


def _normalise_builder(builder: str) -> str:
    aliases = {
        "global": "hierarchical_complete",
        "global_complete": "hierarchical_complete",
        "global_single": "hierarchical_single",
        "linear": "linear_quantile",
        "linear_baseline": "linear_quantile",
    }
    return aliases.get(builder, builder)


def _pick_best(rows: Sequence[dict[str, str]], *, score: str) -> dict[str, float | str]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("k", ""),
            row.get("dv", ""),
            row.get("projection_mode", ""),
            _normalise_builder(row.get("macro_builder", "")),
            row.get("macro_eps", ""),
        )
        grouped.setdefault(key, []).append(row)

    if not grouped:
        raise ValueError("No rows found in runs.csv.")

    best_key = None
    best_tuple = (math.inf, math.inf, math.inf, math.inf, math.inf)
    best_summary: dict[str, float | str] = {}

    for key, items in grouped.items():
        macro_mean = _mean([_safe_float(item.get("macro_logloss")) for item in items], default=math.nan)
        pred_mean = _mean([_safe_float(item.get("selection_predictive")) for item in items], default=math.nan)
        st_mean = _mean([_safe_float(item.get("selection_stability")) for item in items], default=math.nan)
        ll_mean = _mean([_safe_float(item.get("gaussian_logloss")) for item in items], default=math.nan)
        if not math.isfinite(ll_mean):
            continue
        score_mean = macro_mean if score == "predictive" else st_mean
        if score == "predictive" and not math.isfinite(score_mean):
            score_mean = pred_mean if math.isfinite(pred_mean) else ll_mean
        if score == "stability" and not math.isfinite(score_mean):
            score_mean = -math.inf
        k_val = _safe_float(key[0])
        dv_val = _safe_float(key[1])
        if score == "predictive":
            rank = (
                score_mean,
                -st_mean if math.isfinite(st_mean) else math.inf,
                ll_mean,
                k_val if math.isfinite(k_val) else math.inf,
                dv_val if math.isfinite(dv_val) else math.inf,
            )
        else:
            rank = (
                -score_mean,
                pred_mean if math.isfinite(pred_mean) else ll_mean,
                ll_mean,
                k_val if math.isfinite(k_val) else math.inf,
                dv_val if math.isfinite(dv_val) else math.inf,
            )
        if rank < best_tuple:
            best_tuple = rank
            best_key = key
            best_summary = {
                "k": int(k_val),
                "dv": int(dv_val),
                "projection_mode": key[2],
                "macro_builder": key[3],
                "macro_eps": float(_safe_float(key[4])),
                "gaussian_logloss_mean": ll_mean,
                "macro_logloss_mean": macro_mean,
                "selection_predictive_mean": pred_mean,
                "selection_stability_mean": st_mean,
                "selection_objective_mean": score_mean,
                "selection_objective": score,
                "n_rows": len(items),
            }

    if best_key is None:
        raise ValueError("Could not identify a valid best model from runs.csv.")
    return best_summary


def _write_tractability_report(path: Path, rows: Sequence[dict[str, str]]) -> None:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        builder = _normalise_builder(row.get("macro_builder", ""))
        groups.setdefault(builder, []).append(row)

    report_rows: list[dict[str, float | int | str]] = []
    for builder, items in sorted(groups.items()):
        n_pred_vals = [_safe_float(item.get("macro_n_pred")) for item in items]
        mem_vals = [_safe_float(item.get("macro_distance_matrix_mb_est")) for item in items]
        time_vals = [_safe_float(item.get("macro_build_time_s")) for item in items]
        macro_vals = [_safe_float(item.get("macro_logloss")) for item in items]
        ll_vals = [_safe_float(item.get("gaussian_logloss")) for item in items]
        report_rows.append(
            {
                "macro_builder": builder,
                "n_runs": len(items),
                "N_mean": _mean(n_pred_vals),
                "N_max": _max(n_pred_vals),
                "distance_matrix_mb_mean": _mean(mem_vals, default=0.0),
                "distance_matrix_mb_max": _max(mem_vals, default=0.0),
                "macro_build_time_s_mean": _mean(time_vals),
                "macro_build_time_s_max": _max(time_vals),
                "macro_logloss_mean": _mean(macro_vals),
                "gaussian_logloss_mean": _mean(ll_vals),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "macro_builder",
                "n_runs",
                "N_mean",
                "N_max",
                "distance_matrix_mb_mean",
                "distance_matrix_mb_max",
                "macro_build_time_s_mean",
                "macro_build_time_s_max",
                "macro_logloss_mean",
                "gaussian_logloss_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-run continuous PRISM from a file and select the best configuration."
    )
    parser.add_argument("data_path", type=Path, help="Path to .csv/.txt/.npy series file.")
    parser.add_argument("--data-column", type=int, default=None, help="Single column index.")
    parser.add_argument("--data-columns", nargs="+", type=int, default=None, help="Multiple column indices.")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--length", type=int, default=None, help="Optional prefix length to use from the file.")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Outer train fraction.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds for repeated runs.")
    parser.add_argument(
        "--score",
        type=str,
        choices=["predictive", "stability"],
        default="predictive",
        help="Objective used for model selection.",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=None, help="Optional latent dimension grid.")
    parser.add_argument("--dvs", nargs="+", type=int, default=None, help="Optional macro dimension grid.")
    parser.add_argument("--max-d", type=int, default=6, help="Max latent dimension when --ks is omitted.")
    parser.add_argument("--max-dv", type=int, default=3, help="Max macro dimension when --dvs is omitted.")
    parser.add_argument("--em-iters", type=int, default=25, help="EM iterations.")
    parser.add_argument("--macro-bins", type=int, default=3, help="Quantile bins per macro dimension.")
    parser.add_argument(
        "--selection-train-frac",
        type=float,
        default=0.75,
        help="Inner blocked train fraction for model selection.",
    )
    parser.add_argument(
        "--selection-projections",
        nargs="+",
        type=str,
        choices=["pca", "random", "psi_opt"],
        default=["pca", "random", "psi_opt"],
        help="Projection families explored in auto selection.",
    )
    parser.add_argument(
        "--selection-builders",
        nargs="+",
        type=str,
        choices=[
            "greedy",
            "hierarchical_single",
            "hierarchical_complete",
            "linear_quantile",
            "global",
            "global_single",
            "global_complete",
            "linear",
        ],
        default=["hierarchical_single", "hierarchical_complete", "linear_quantile", "greedy"],
        help="Macro builders explored in auto selection.",
    )
    parser.add_argument(
        "--selection-eps",
        nargs="+",
        type=float,
        default=[0.15, 0.2, 0.25, 0.3, 0.4],
        help="Epsilon grid explored in auto selection.",
    )
    parser.add_argument("--selection-repeats", type=int, default=3, help="Repeated fits for stability scoring.")
    parser.add_argument("--selection-seed-stride", type=int, default=97, help="Seed increment between repeats.")
    parser.add_argument(
        "--selection-perturb",
        type=str,
        choices=["seed", "blocked", "seed_blocked"],
        default="seed_blocked",
        help="Perturbation family used in model-selection repeats.",
    )
    parser.add_argument(
        "--selection-block-frac",
        type=float,
        default=0.85,
        help="Blocked-resample fraction for model-selection perturbations.",
    )
    parser.add_argument("--psi-restarts", type=int, default=4, help="Psi restarts used in auto mode.")
    parser.add_argument("--psi-iters", type=int, default=40, help="Psi iterations per restart used in auto mode.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging(getattr(logging, args.log_level))

    matrix = load_timeseries(
        args.data_path,
        column=args.data_column,
        columns=tuple(args.data_columns) if args.data_columns is not None else None,
    )
    full_len = int(matrix.shape[0])
    obs_dim = int(matrix.shape[1])
    length = full_len if args.length is None else int(args.length)
    if length < 50:
        raise ValueError("length must be at least 50 for stable ISS fitting.")
    if length > full_len:
        raise ValueError(f"Requested length={length}, but dataset only has {full_len} rows.")
    if not (0.55 <= args.train_frac < 1.0):
        raise ValueError("train-frac must lie in [0.55, 1.0).")

    split = int(length * args.train_frac)
    split = max(1, min(split, length - 1))
    train_len = split
    ks = args.ks if args.ks is not None else _default_d_grid(train_len, obs_dim, max_d=args.max_d)
    dvs = args.dvs if args.dvs is not None else _default_dv_grid(obs_dim, max_dv=args.max_dv)
    if any(k < 1 for k in ks):
        raise ValueError("All ks must be >= 1.")
    if any(dv < 1 for dv in dvs):
        raise ValueError("All dvs must be >= 1.")
    if any(dv > obs_dim for dv in dvs):
        raise ValueError(f"All dvs must be <= observation dimension p={obs_dim}.")
    if args.selection_repeats < 1:
        raise ValueError("selection-repeats must be >= 1.")
    if args.selection_seed_stride < 1:
        raise ValueError("selection-seed-stride must be >= 1.")
    if not (0.55 <= args.selection_block_frac <= 1.0):
        raise ValueError("selection-block-frac must lie in [0.55, 1.0].")
    seen_builders: set[str] = set()
    sel_builders_list: list[str] = []
    for name in args.selection_builders:
        builder = _normalise_builder(name)
        if builder in seen_builders:
            continue
        seen_builders.add(builder)
        sel_builders_list.append(builder)
    sel_builders = tuple(sel_builders_list)

    run_id = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    outdir = args.outdir or Path("results") / f"auto_{args.data_path.stem}_{run_id}"
    runs_csv = outdir / "runs.csv"
    if runs_csv.exists() and not args.force:
        raise FileExistsError(f"{outdir} already exists; pass --force to overwrite.")
    if runs_csv.exists() and args.force:
        runs_csv.unlink()
    outdir.mkdir(parents=True, exist_ok=True)

    process = ContinuousFile(
        path=args.data_path,
        column=args.data_column,
        columns=tuple(args.data_columns) if args.data_columns is not None else None,
    )
    recon = KalmanISSReconstructor(
        em_iters=args.em_iters,
        macro_bins=args.macro_bins,
        model_select=True,
        selection_score=args.score,
        selection_train_frac=args.selection_train_frac,
        selection_projection_modes=tuple(args.selection_projections),
        selection_macro_builders=sel_builders,
        selection_macro_eps=tuple(args.selection_eps),
        selection_repeats=args.selection_repeats,
        selection_seed_stride=args.selection_seed_stride,
        selection_perturb=args.selection_perturb,
        selection_block_frac=args.selection_block_frac,
        macro_builder="hierarchical_complete",
        projection_mode="pca",
        macro_eps=0.25,
        psi_restarts=args.psi_restarts,
        psi_iterations=args.psi_iters,
    )
    reps = [ISSDim(d=d, dv=dv) for d in ks for dv in dvs]

    run_experiment(
        process=process,
        reconstructor=recon,
        representations=reps,
        length=length,
        train_frac=args.train_frac,
        seeds=args.seeds,
        outdir=outdir,
    )
    summarise(outdir)

    rows = _parse_rows(outdir / "runs.csv")
    best = _pick_best(rows, score=args.score)
    save_json(outdir / "best_selection.json", best)

    best_outdir = outdir / "best_pipeline"
    best_runs_csv = best_outdir / "runs.csv"
    if best_runs_csv.exists():
        best_runs_csv.unlink()
    best_outdir.mkdir(parents=True, exist_ok=True)

    best_recon = KalmanISSReconstructor(
        em_iters=args.em_iters,
        macro_bins=args.macro_bins,
        model_select=False,
        projection_mode=str(best["projection_mode"]),
        macro_builder=str(best["macro_builder"]),
        macro_eps=float(best["macro_eps"]),
        psi_restarts=args.psi_restarts,
        psi_iterations=args.psi_iters,
    )
    best_rep = [ISSDim(d=int(best["k"]), dv=int(best["dv"]))]
    run_experiment(
        process=process,
        reconstructor=best_recon,
        representations=best_rep,
        length=length,
        train_frac=args.train_frac,
        seeds=args.seeds,
        outdir=best_outdir,
    )
    summarise(best_outdir)

    tract_root = outdir / "tractability"
    tract_root.mkdir(parents=True, exist_ok=True)
    builder_order = list(sel_builders)
    if "linear_quantile" not in builder_order:
        builder_order.append("linear_quantile")
    track_rows: list[dict[str, str]] = []
    for builder in builder_order:
        builder_outdir = tract_root / builder
        builder_runs = builder_outdir / "runs.csv"
        if builder_runs.exists():
            builder_runs.unlink()
        builder_outdir.mkdir(parents=True, exist_ok=True)
        track_recon = KalmanISSReconstructor(
            em_iters=args.em_iters,
            macro_bins=args.macro_bins,
            model_select=False,
            projection_mode=str(best["projection_mode"]),
            macro_builder=builder,
            macro_eps=float(best["macro_eps"]),
            psi_restarts=args.psi_restarts,
            psi_iterations=args.psi_iters,
        )
        run_experiment(
            process=process,
            reconstructor=track_recon,
            representations=best_rep,
            length=length,
            train_frac=args.train_frac,
            seeds=args.seeds,
            outdir=builder_outdir,
        )
        track_rows.extend(_parse_rows(builder_outdir / "runs.csv"))
    _write_tractability_report(outdir / "tractability_by_builder.csv", track_rows)

    save_json(
        outdir / "auto_config.json",
        {
            "data_path": str(args.data_path),
            "length": length,
            "obs_dim": obs_dim,
            "train_frac": args.train_frac,
            "seeds": args.seeds,
            "ks": ks,
            "dvs": dvs,
            "selection_score": args.score,
            "selection_projections": args.selection_projections,
            "selection_builders": list(sel_builders),
            "selection_eps": args.selection_eps,
            "selection_perturb": args.selection_perturb,
            "selection_block_frac": args.selection_block_frac,
            "best": best,
        },
    )
    print(f"Auto sweep complete: {outdir}")
    print(f"Best pipeline run: {best_outdir}")


if __name__ == "__main__":
    main()
