"""Plot summary metrics as a function of representation size `k`."""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_int(text: str) -> Optional[int]:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _safe_float(text: str) -> Optional[float]:
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _matches_any_float(value: float, targets: Optional[Sequence[float]], tol: float = 1e-12) -> bool:
    if targets is None:
        return True
    return any(abs(value - target) <= tol for target in targets)


def filter_rows(
    rows: List[Dict[str, str]],
    subsample_step: int,
    base_process: Optional[str] = None,
    representation: Optional[str] = None,
    flip_ps: Optional[Sequence[float]] = None,
    condition_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        step = _safe_int(row.get("subsample_step", ""))
        if step is None or step != subsample_step:
            continue
        if base_process is not None and row.get("base_process", "") != base_process:
            continue
        if representation is not None and row.get("representation", "") != representation:
            continue
        if condition_id is not None and row.get("condition_id", "") != condition_id:
            continue
        flip_p = _safe_float(row.get("flip_p", "0.0"))
        if flip_p is None:
            continue
        if not _matches_any_float(flip_p, flip_ps):
            continue
        out.append(row)
    return out


def plot_metric_vs_k(rows: List[Dict[str, str]], metric: str, outpath: Path) -> bool:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if not rows or mean_col not in rows[0]:
        print(f"Skipping {metric}: column {mean_col} not available.")
        return False

    by_flip: Dict[float, List[tuple[int, float, float]]] = {}
    for row in rows:
        flip = _safe_float(row.get("flip_p", ""))
        k = _safe_int(row.get("k", ""))
        mean_value = _safe_float(row.get(mean_col, ""))
        std_value = _safe_float(row.get(std_col, "0") or "0")
        if flip is None or k is None or mean_value is None:
            continue
        by_flip.setdefault(flip, []).append((k, mean_value, 0.0 if std_value is None else std_value))

    if not by_flip:
        print(f"Skipping {metric}: no finite values after filtering.")
        return False

    plt.figure(figsize=(4.8, 3.2))
    any_curve = False
    for flip, points in sorted(by_flip.items(), key=lambda item: item[0]):
        points_sorted = sorted(points, key=lambda item: item[0])
        xs = [point[0] for point in points_sorted]
        ys = [point[1] for point in points_sorted]
        yerrs = [point[2] for point in points_sorted]
        if not xs:
            continue
        any_curve = True
        plt.errorbar(xs, ys, yerr=yerrs, fmt="o-", capsize=3, label=f"flip_p={flip:g}")
    if not any_curve:
        plt.close()
        print(f"Skipping {metric}: no plottable points.")
        return False

    plt.xlabel("Representation size")
    ylabel = {
        "branch_entropy": "Branching entropy (bits)",
        "unifilarity_score": "Unifilarity score",
        "logloss": "Log-loss (nats)",
        "gaussian_logloss": "Gaussian log-loss (nats)",
        "n_states": "State count / latent dimension",
        "C_mu_empirical": "$C_\\mu$ (empirical)",
    }.get(metric, metric)
    plt.ylabel(ylabel)
    if len(by_flip) > 1:
        plt.legend(fontsize=8)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Wrote {outpath}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subsample-step", type=int, default=1)
    parser.add_argument("--base-process", type=str, default=None)
    parser.add_argument("--representation", type=str, default=None)
    parser.add_argument("--flip-ps", nargs="+", type=float, default=None)
    parser.add_argument("--condition-id", type=str, default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["branch_entropy", "unifilarity_score", "logloss"],
        help="Metrics to plot.",
    )
    args = parser.parse_args()

    summary = args.root / "summary_by_condition.csv"
    if not summary.exists():
        raise FileNotFoundError(f"Missing {summary}. Run summarise first.")
    rows = filter_rows(
        read_rows(summary),
        subsample_step=args.subsample_step,
        base_process=args.base_process,
        representation=args.representation,
        flip_ps=args.flip_ps,
        condition_id=args.condition_id,
    )
    if not rows:
        raise ValueError("No rows match the provided filters.")

    figs = args.root / "figures"
    figs.mkdir(exist_ok=True)
    for metric in args.metrics:
        outpath = figs / f"{metric}_vs_k_sub{args.subsample_step}.png"
        plot_metric_vs_k(rows, metric=metric, outpath=outpath)


if __name__ == "__main__":
    main()
