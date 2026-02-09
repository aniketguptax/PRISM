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


def read_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "flip_p",
            "subsample_step",
            "k",
            "branch_entropy_mean",
            "unifilarity_score_mean",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        for r in reader:
            rows.append(
                Row(
                    flip_p=float(r["flip_p"]),
                    subsample_step=int(float(r["subsample_step"])),
                    k=int(float(r["k"])),
                    branch_mean=float(r["branch_entropy_mean"]),
                    unif_mean=float(r["unifilarity_score_mean"]),
                    base_process=r.get("base_process", ""),
                    representation=r.get("representation", ""),
                    condition_id=r.get("condition_id", ""),
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
    for r in rows:
        if base_process is not None and r.base_process != base_process:
            continue
        if representation is not None and r.representation != representation:
            continue
        if condition_id is not None and r.condition_id != condition_id:
            continue
        out.append(r)
    return out

def build_grid(rows: List[Row]) -> Tuple[List[int], List[float], List[List[float]], List[List[float]]]:
    ks = sorted(set(r.k for r in rows))
    flip_ps = sorted(set(r.flip_p for r in rows))

    k_index: Dict[int, int] = {k: i for i, k in enumerate(ks)}
    f_index: Dict[float, int] = {fp: j for j, fp in enumerate(flip_ps)}

    branch_grid = [[math.nan for _ in ks] for _ in flip_ps]
    omu_grid = [[math.nan for _ in ks] for _ in flip_ps]

    for r in rows:
        j = f_index[r.flip_p]
        i = k_index[r.k]
        branch_grid[j][i] = r.branch_mean
        omu_grid[j][i] = 1.0 - r.unif_mean

    return ks, flip_ps, branch_grid, omu_grid

def make_closed_mask(
    ks: List[int],
    flip_ps: List[float],
    branch_grid: List[List[float]],
    omu_grid: List[List[float]],
    unif_threshold: float,
    branch_threshold: float,
) -> List[List[bool]]:
    # omu_grid = 1 - unif, so unif >= T  <=>  omu <= 1-T
    omu_thresh = 1.0 - unif_threshold
    mask: List[List[bool]] = [[False for _ in ks] for _ in flip_ps]
    
    for j in range(len(flip_ps)):
        for i in range(len(ks)):
            b = branch_grid[j][i]
            omu = omu_grid[j][i]
            if math.isnan(b) or math.isnan(omu):
                continue
            mask[j][i] = (omu <= omu_thresh) and (b <= branch_threshold)
    return mask

def plot_heatmap(
    ks: List[int],
    flip_ps: List[float],
    grid: List[List[float]],
    title: str,
    ylabel: str,
    outpath: Path,
    closed_mask: Optional[List[List[bool]]] = None,
) -> None:
    plt.figure(figsize=(6.2, 3.6))

    im = plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            float(min(ks) - 0.5),
            float(max(ks) + 0.5),
            float(min(flip_ps)),
            float(max(flip_ps)),
        ),
    )
    plt.colorbar(im)

    plt.xlabel("Representation length $k$")
    plt.ylabel(ylabel)
    plt.title(title)

    # Overlay closure mask as points (centres)
    if closed_mask is not None:
        xs: List[float] = []
        ys: List[float] = []
        for j, fp in enumerate(flip_ps):
            for i, k in enumerate(ks):
                if closed_mask[j][i]:
                    xs.append(k)
                    ys.append(fp)
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



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing results and figures")
    parser.add_argument("--infile", type=str, default="summary_by_condition.csv", help="Input CSV file name")
    parser.add_argument("--unif-threshold", type=float, default=0.98, help="Unifilarity threshold")
    parser.add_argument("--branch-threshold", type=float, default=0.02, help="Branching entropy threshold (bits)")
    parser.add_argument("--process-name", type=str, default=None, help="Name of the process (for plot titles)")
    
    parser.add_argument("--ylabel", type=str, default="flip probability $p$", help="Y-axis label for plots")
    parser.add_argument("--metric", choices=["branch", "omu", "both"], default="both", help="Which metric(s) to plot")
    
    # fiter options
    parser.add_argument("--base-process", type=str, default=None, help="Name of the base process to filter")
    parser.add_argument("--representation", type=str, default=None, help="Name of the representation to filter")
    parser.add_argument("--condition-id", type=str, default=None, help="Condition ID to filter")
    
    args = parser.parse_args()

    summary_csv = args.root / args.infile
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing {summary_csv}")

    all_rows = read_rows(summary_csv)
    all_rows = filter_rows(
        all_rows,
        base_process=args.base_process,
        representation=args.representation,
        condition_id=args.condition_id,
    )

    # group by subsample step
    steps = sorted(set(r.subsample_step for r in all_rows))
    figs_dir = args.root / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Use process name in titles if provided, else leave generic
    process_prefix = f"{args.process_name}: " if args.process_name else ""

    for step in steps:
        rows = [r for r in all_rows if r.subsample_step == step]
        if not rows:
            print(f"No data for subsample step {step}, skipping.")
            continue
        
        ks, flip_ps, branch_grid, omu_grid = build_grid(rows)

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