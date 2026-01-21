import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


FOLDER_RE = re.compile(
    r"""
    (?P<base>.+?)                              # base process name (greedy, until suffixes)
    (?:_flip(?P<flip>[0-9]+(?:\.[0-9]+)?))?     # optional _flip0.05
    (?:_sub(?P<sub>[0-9]+))?                   # optional _sub2
    $""",
    re.VERBOSE,
)


def parse_condition_from_folder(folder_name: str) -> Tuple[str, float, int]:
    m = FOLDER_RE.match(folder_name)
    if not m:
        # fallback: treat as base only
        return folder_name, 0.0, 1

    base = m.group("base")
    flip_s = m.group("flip")
    sub_s = m.group("sub")

    flip_p = float(flip_s) if flip_s is not None else 0.0
    sub = int(sub_s) if sub_s is not None else 1
    return base, flip_p, sub


def extract_k(rep_name: str) -> Optional[int]:
    # expects "last_3" or "last_3_noisy" (ignore suffixes)
    m = re.search(r"last_(\d+)", rep_name)
    return int(m.group(1)) if m else None


def find_metrics_files(root: Path) -> List[Path]:
    return sorted(root.rglob("metrics.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Grid sweep root folder, e.g. results/grid_even_v1")
    parser.add_argument("--outdir", type=Path, default=None, help="Where to write outputs (default: <root>)")
    args = parser.parse_args()

    root = args.root
    outdir = args.outdir or root
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_files = find_metrics_files(root)
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.csv found under {root}")

    rows = []
    for mf in metrics_files:
        # folder containing metrics.csv is the condition folder
        condition_folder = mf.parent
        base, flip_p, sub = parse_condition_from_folder(condition_folder.name)

        df = pd.read_csv(mf)

        # Add inferred condition columns
        df["condition_folder"] = condition_folder.name
        df["base_process_variant"] = base
        df["flip_p"] = flip_p
        df["subsample_step"] = sub

        # Extract k from representation
        df["k"] = df["representation"].apply(extract_k)

        # Guard: drop rows with no k (shouldn’t happen, but keeps things robust)
        df = df[df["k"].notna()].copy()
        df["k"] = df["k"].astype(int)

        rows.append(df)

    all_runs = pd.concat(rows, ignore_index=True)

    # Write raw concatenated runs
    all_path = outdir / "all_runs.csv"
    all_runs.to_csv(all_path, index=False)
    print(f"Wrote {all_path}")

    # Aggregate over seeds (and any repeats) within each condition + k
    metrics_to_agg = [
        "logloss",
        "n_states",
        "C_mu_empirical",
        "unifilarity_score",
        "branch_entropy",
    ]
    metrics_to_agg = [m for m in metrics_to_agg if m in all_runs.columns]

    group_cols = ["base_process_variant", "flip_p", "subsample_step", "k", "process", "reconstructor", "representation"]
    grouped = all_runs.groupby(group_cols, dropna=False)

    summary = grouped[metrics_to_agg].agg(["mean", "std"]).reset_index()

    # Flatten MultiIndex columns from agg
    summary.columns = [
        f"{a}_{b}" if b else a
        for (a, b) in [(c[0], c[1] if isinstance(c, tuple) else "") if isinstance(c, tuple) else (c, "") for c in summary.columns]
    ]

    # Cleaner manual flattening, rebuild columns properly:
    clean_cols = []
    for col in summary.columns:
        if isinstance(col, tuple):
            base, stat = col
            clean_cols.append(base if stat == "" else f"{base}_{stat}")
        else:
            clean_cols.append(col)
    summary.columns = clean_cols

    summary_path = outdir / "summary_by_condition.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    # Convenience: also write a “simple summary” for the most used metrics
    wanted = []
    for m in ["branch_entropy", "unifilarity_score", "logloss"]:
        if f"{m}_mean" in summary.columns:
            wanted += [f"{m}_mean", f"{m}_std"]
    simple_cols = ["base_process_variant", "flip_p", "subsample_step", "k"] + wanted
    simple = summary[simple_cols].sort_values(["flip_p", "subsample_step", "k"])

    simple_path = outdir / "summary_simple.csv"
    simple.to_csv(simple_path, index=False)
    print(f"Wrote {simple_path}")


if __name__ == "__main__":
    main()