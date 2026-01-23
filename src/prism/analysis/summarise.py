import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

METRICS = ["logloss", "n_states", "C_mu_empirical", "unifilarity_score", "branch_entropy"]

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

def std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def summarise(run_root: Path) -> None:
    runs_path = run_root / "runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing {runs_path}")

    key_cols = ["base_process", "condition_id", "flip_p", "subsample_step", "reconstructor", "representation", "k"]

    with runs_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = set(key_cols) | {"seed"} | set(METRICS)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {runs_path}: {sorted(missing)}")
        rows = list(reader)

    
    groups: Dict[Tuple[Any, ...], List[Dict[str, str]]] = defaultdict(list)
    
    for r in rows:
        key = tuple(r.get(c, "") for c in key_cols)
        groups[key].append(r)

    out_rows = []
    for key, rs in groups.items():
        out = {c: v for c, v in zip(key_cols, key)}
        out["n"] = len(rs)
        for m in METRICS:
            if m not in rs[0]:
                continue
            xs = [float(r[m]) for r in rs if r.get(m, "") != ""]
            out[f"{m}_mean"] = mean(xs)
            out[f"{m}_std"] = std(xs)
        out_rows.append(out)

    out_rows.sort(key=lambda r: (float(r["flip_p"]), int(r["subsample_step"]), int(r["k"] or 0)))

    summary_path = run_root / "summary_by_condition.csv"
    fieldnames = list(out_rows[0].keys()) if out_rows else key_cols
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    simple_cols = ["base_process", "flip_p", "subsample_step", "k",
                   "logloss_mean","logloss_std",
                   "unifilarity_score_mean","unifilarity_score_std",
                   "branch_entropy_mean","branch_entropy_std"]
    simple_path = run_root / "summary_simple.csv"
    with simple_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=simple_cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in simple_cols})

def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Run root folder containing runs.csv",
    )
    
    args = parser.parse_args()
    summarise(args.root)

if __name__ == "__main__":
    main()