import csv
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt

def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _as_int(s: str) -> int:
    return int(float(s))

def _as_float(s: str) -> float:
    return float(s)

def _matches_any_float(value: float, targets: Optional[Sequence[float]], tol: float = 1e-12) -> bool:
    if targets is None:
        return True
    return any(abs(value - t) <= tol for t in targets)

def filter_rows(
    rows: List[Dict[str, str]],
    subsample_step: int,
    base_process: Optional[str] = None,
    representation: Optional[str] = None,
    flip_ps: Optional[Sequence[float]] = None,
    condition_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    
    for r in rows:
        if "subsample_step" not in r:
            raise ValueError("Missing 'subsample_step' column in rows")
        if _as_int(r["subsample_step"]) != subsample_step:
            continue
        if base_process is not None and r.get("base_process", "") != base_process:
            continue
        if representation is not None and r.get("representation", "") != representation:
            continue
        if condition_id is not None and r.get("condition_id", "") != condition_id:
            continue
        
        if "flip_p" in r:
            flip_p = _as_float(r["flip_p"])
            if not _matches_any_float(flip_p, flip_ps):
                continue
        
        out.append(r)
    return out
            
def plot_metric_vs_k(
    rows: List[Dict[str, str]],
    metric: str,
    outpath: Path,
) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if not rows:
        raise ValueError(f"No rows after filtering to plot {metric} vs k")
    if mean_col not in rows[0]:
        # show what columns are available
        available = sorted(rows[0].keys())
        raise ValueError(f"Missing column '{mean_col}' in rows. Available columns: {available}")

    # group by flip_p
    by_flip: Dict[float, List[Dict[str, str]]] = {}
    for r in rows:
        flip = _as_float(r.get("flip_p", "0.0"))
        by_flip.setdefault(flip, []).append(r)

    plt.figure(figsize=(4.8, 3.2))

    for flip, rs in sorted(by_flip.items(), key=lambda x: x[0]):
        rs = sorted(rs, key=lambda r: _as_int(r["k"]))
        x = [_as_int(r["k"]) for r in rs]
        y = [float(r[mean_col]) for r in rs]
        yerr = [float(r.get(std_col, 0.0) or 0.0) for r in rs]
        plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, label=f"flip_p={flip:g}")

    plt.xlabel("Representation length $k$")
    ylabel = {
        "branch_entropy": "Branching entropy (bits)",
        "unifilarity_score": "Unifilarity score",
        "logloss": "Log-loss (nats)",
        "n_states": "Number of states",
        "C_mu_empirical": "$C_\\mu$ (empirical)"
    }.get(metric, metric)
    plt.ylabel(ylabel)

    # legend if multiple flip_ps
    if len(by_flip) > 1:
        plt.legend(fontsize=8)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Wrote {outpath}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--subsample-step", type=int, default=1)
    
    # optional filters
    p.add_argument("--base-process", type=str, default=None, help="Name of the base process to filter")
    p.add_argument("--representation", type=str, default=None, help="Name of the representation to filter")
    p.add_argument("--flip-ps", nargs="+", type=float, default=None, help="Only plot these flip probabilities")
    p.add_argument("--condition-id", type=str, default=None, help="Condition ID to filter")
    
    # what to plot
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["branch_entropy","unifilarity_score"],
        help="Metrics to plot"
    )
    
    args = p.parse_args()

    summary = args.root / "summary_by_condition.csv"
    if not summary.exists():
        raise FileNotFoundError(f"Missing {summary}. Run summarise first.")

    rows = read_rows(summary)
    rows = filter_rows(
        rows,
        subsample_step=args.subsample_step,
        base_process=args.base_process,
        representation=args.representation,
        flip_ps=args.flip_ps,
        condition_id=args.condition_id,
    )
    
    figs = args.root / "figures"
    figs.mkdir(exist_ok=True)

    for metric in args.metrics:
        outpath = figs / f"{metric}_vs_k_sub{args.subsample_step}.png"
        plot_metric_vs_k(rows, metric=metric, outpath=outpath)

if __name__ == "__main__":
    main()