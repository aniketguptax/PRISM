"""Aggregate per-seed PRISM run metrics into summary tables."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_METRICS = [
    "logloss",
    "gaussian_logloss",
    "n_states",
    "C_mu_empirical",
    "unifilarity_score",
    "branch_entropy",
]


def _safe_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def summarise(run_root: Path) -> None:
    runs_path = run_root / "runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing {runs_path}")

    key_cols = ["base_process", "condition_id", "flip_p", "subsample_step", "reconstructor", "representation", "k"]
    with runs_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{runs_path} has no header row.")
        required_columns = {"seed"} | set(key_cols)
        missing_keys = required_columns - set(reader.fieldnames)
        if missing_keys:
            raise ValueError(f"Missing required columns in {runs_path}: {sorted(missing_keys)}")
        rows = list(reader)

    metrics = [metric for metric in DEFAULT_METRICS if metric in rows[0]] if rows else []
    groups: Dict[Tuple[Any, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(col, "") for col in key_cols)
        groups[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, grouped_rows in groups.items():
        out: dict[str, Any] = {col: value for col, value in zip(key_cols, key)}
        out["n"] = len(grouped_rows)
        for metric in metrics:
            values = [_safe_float(r.get(metric, "")) for r in grouped_rows]
            finite_values = [v for v in values if v is not None]
            out[f"{metric}_mean"] = _mean(finite_values)
            out[f"{metric}_std"] = _std(finite_values)
        summary_rows.append(out)

    def _key_sort(row: dict[str, Any]) -> Tuple[float, int, int]:
        flip = _safe_float(str(row.get("flip_p", "0")))
        step = _safe_float(str(row.get("subsample_step", "1")))
        kval = _safe_float(str(row.get("k", "0")))
        return (
            0.0 if flip is None else flip,
            1 if step is None else int(step),
            0 if kval is None else int(kval),
        )

    summary_rows.sort(key=_key_sort)

    summary_path = run_root / "summary_by_condition.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
    else:
        fieldnames = key_cols + ["n"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    simple_cols = [
        "base_process",
        "flip_p",
        "subsample_step",
        "k",
        "logloss_mean",
        "logloss_std",
        "gaussian_logloss_mean",
        "gaussian_logloss_std",
        "unifilarity_score_mean",
        "unifilarity_score_std",
        "branch_entropy_mean",
        "branch_entropy_std",
    ]
    simple_path = run_root / "summary_simple.csv"
    with simple_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=simple_cols)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({column: row.get(column, "") for column in simple_cols})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Run root folder containing runs.csv")
    args = parser.parse_args()
    summarise(args.root)


if __name__ == "__main__":
    main()
