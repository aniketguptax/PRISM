"""Phase-diagram plots for closure metrics over `(flip_p, k)`."""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle


@dataclass(frozen=True)
class Row:
    flip_p: float
    subsample_step: int
    k: int
    branch_mean: float
    unif_mean: float
    base_process: str
    representation: str
    condition_id: str


def _safe_float(value: str) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"flip_p", "subsample_step", "k", "branch_entropy_mean", "unifilarity_score_mean"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for raw in reader:
            flip = _safe_float(raw.get("flip_p", ""))
            step = _safe_int(raw.get("subsample_step", ""))
            kval = _safe_int(raw.get("k", ""))
            branch = _safe_float(raw.get("branch_entropy_mean", ""))
            unif = _safe_float(raw.get("unifilarity_score_mean", ""))
            if flip is None or step is None or kval is None or branch is None or unif is None:
                continue
            rows.append(
                Row(
                    flip_p=flip,
                    subsample_step=step,
                    k=kval,
                    branch_mean=branch,
                    unif_mean=unif,
                    base_process=raw.get("base_process", ""),
                    representation=raw.get("representation", ""),
                    condition_id=raw.get("condition_id", ""),
                )
            )
    return rows


def filter_rows(
    rows: List[Row],
    base_process: Optional[str],
    representation: Optional[str],
    condition_id: Optional[str],
) -> List[Row]:
    out: List[Row] = []
    for row in rows:
        if base_process is not None and row.base_process != base_process:
            continue
        if representation is not None and row.representation != representation:
            continue
        if condition_id is not None and row.condition_id != condition_id:
            continue
        out.append(row)
    return out


def build_grid(rows: List[Row]) -> Tuple[List[int], List[float], List[List[float]], List[List[float]]]:
    ks = sorted(set(row.k for row in rows))
    flip_ps = sorted(set(row.flip_p for row in rows))
    k_index: Dict[int, int] = {k: idx for idx, k in enumerate(ks)}
    flip_index: Dict[float, int] = {flip: idx for idx, flip in enumerate(flip_ps)}
    branch_grid = [[math.nan for _ in ks] for _ in flip_ps]
    omu_grid = [[math.nan for _ in ks] for _ in flip_ps]
    for row in rows:
        j = flip_index[row.flip_p]
        i = k_index[row.k]
        branch_grid[j][i] = row.branch_mean
        omu_grid[j][i] = 1.0 - row.unif_mean
    return ks, flip_ps, branch_grid, omu_grid


def make_closed_mask(
    ks: List[int],
    flip_ps: List[float],
    branch_grid: List[List[float]],
    omu_grid: List[List[float]],
    unif_threshold: float,
    branch_threshold: float,
) -> List[List[bool]]:
    omu_threshold = 1.0 - unif_threshold
    mask: List[List[bool]] = [[False for _ in ks] for _ in flip_ps]
    for j in range(len(flip_ps)):
        for i in range(len(ks)):
            branch = branch_grid[j][i]
            omu = omu_grid[j][i]
            if math.isnan(branch) or math.isnan(omu):
                continue
            mask[j][i] = (omu <= omu_threshold) and (branch <= branch_threshold)
    return mask


def plot_heatmap(
    ks: List[int],
    flip_ps: List[float],
    grid: List[List[float]],
    title: str,
    ylabel: str,
    outpath: Path,
    closed_mask: Optional[List[List[bool]]] = None,
) -> bool:
    finite_values = [value for row in grid for value in row if not math.isnan(value)]
    if not finite_values:
        print(f"Skipping {title}: all values are NaN.")
        return False

    plt.figure(figsize=(6.2, 3.6))
    x_min = float(min(ks) - 0.5)
    x_max = float(max(ks) + 0.5)
    y_min = float(min(flip_ps))
    y_max = float(max(flip_ps))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01
    im = plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(x_min, x_max, y_min, y_max),
    )
    plt.colorbar(im)
    plt.xlabel("Representation size")
    plt.ylabel(ylabel)
    plt.title(title)

    if closed_mask is not None:
        xs: List[float] = []
        ys: List[float] = []
        for j, flip in enumerate(flip_ps):
            for i, k in enumerate(ks):
                if closed_mask[j][i]:
                    xs.append(k)
                    ys.append(flip)
        if xs:
            plt.scatter(
                xs,
                ys,
                s=12,
                marker=MarkerStyle("o"),
                edgecolors="black",
                facecolors="none",
                linewidths=0.8,
            )

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()
    print(f"Wrote {outpath}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing summary_by_condition.csv")
    parser.add_argument("--infile", type=str, default="summary_by_condition.csv")
    parser.add_argument("--unif-threshold", type=float, default=0.98)
    parser.add_argument("--branch-threshold", type=float, default=0.02)
    parser.add_argument("--process-name", type=str, default=None)
    parser.add_argument("--ylabel", type=str, default="flip probability $p$")
    parser.add_argument("--metric", choices=["branch", "omu", "both"], default="both")
    parser.add_argument("--base-process", type=str, default=None)
    parser.add_argument("--representation", type=str, default=None)
    parser.add_argument("--condition-id", type=str, default=None)
    args = parser.parse_args()

    summary_csv = args.root / args.infile
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing {summary_csv}")

    rows = filter_rows(
        read_rows(summary_csv),
        base_process=args.base_process,
        representation=args.representation,
        condition_id=args.condition_id,
    )
    if not rows:
        print("No finite branch/unifilarity rows after filtering; skipping phase diagram.")
        return

    steps = sorted(set(row.subsample_step for row in rows))
    figs_dir = args.root / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    process_prefix = f"{args.process_name}: " if args.process_name else ""

    for step in steps:
        step_rows = [row for row in rows if row.subsample_step == step]
        if not step_rows:
            continue
        ks, flip_ps, branch_grid, omu_grid = build_grid(step_rows)
        closed_mask = make_closed_mask(
            ks=ks,
            flip_ps=flip_ps,
            branch_grid=branch_grid,
            omu_grid=omu_grid,
            unif_threshold=args.unif_threshold,
            branch_threshold=args.branch_threshold,
        )
        if args.metric in ("branch", "both"):
            plot_heatmap(
                ks,
                flip_ps,
                branch_grid,
                title=f"{process_prefix}mean branching entropy (subsample={step})",
                ylabel=args.ylabel,
                outpath=figs_dir / f"phase_branch_entropy_sub{step}.pdf",
                closed_mask=closed_mask,
            )
        if args.metric in ("omu", "both"):
            plot_heatmap(
                ks,
                flip_ps,
                omu_grid,
                title=f"{process_prefix}mean $1-\\,$unifilarity (subsample={step})",
                ylabel=args.ylabel,
                outpath=figs_dir / f"phase_one_minus_unif_sub{step}.pdf",
                closed_mask=closed_mask,
            )

    print("Done.")


if __name__ == "__main__":
    main()
