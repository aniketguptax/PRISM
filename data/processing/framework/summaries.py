"""Small summary helpers shared by the reporting commands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_METRIC_COLUMNS = ("pred_mse_obs", "pred_r2_obs")


def configure_matplotlib_cache(results_dir: Path) -> tuple[Path, Path]:
    mpl_cache_dir = results_dir / ".mplconfig"
    cache_home_dir = results_dir / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_home_dir.resolve()))
    return mpl_cache_dir, cache_home_dir


def prepare_pyplot(results_dir: Path):
    mpl_cache_dir, cache_home_dir = configure_matplotlib_cache(results_dir)
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_home_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col)
        for col in out.columns
    ]
    return out


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_result_csvs(
    results_dir: Path,
    pattern: str,
    required_columns: Iterable[str],
    *,
    sort_columns: Iterable[str] | None = None,
    categorical_orders: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    files = sorted(results_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No result CSVs matching {pattern!r} found in {results_dir}")

    frames = []
    required_columns = list(required_columns)
    for path in files:
        df = pd.read_csv(path)
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    for column, order in (categorical_orders or {}).items():
        out[column] = pd.Categorical(out[column], order, ordered=True)

    if sort_columns is not None:
        out = out.sort_values(list(sort_columns))
    return out


def compute_metric_means(
    df: pd.DataFrame,
    group_columns: Iterable[str],
    metric_columns: Iterable[str] = DEFAULT_METRIC_COLUMNS,
) -> pd.DataFrame:
    return (
        df.groupby(list(group_columns), observed=False)[list(metric_columns)]
        .mean()
        .reset_index()
    )


def compute_metric_summary(
    df: pd.DataFrame,
    group_columns: Iterable[str],
    metric_columns: Iterable[str] = DEFAULT_METRIC_COLUMNS,
) -> pd.DataFrame:
    summary = (
        df.groupby(list(group_columns), observed=False)[list(metric_columns)]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    return flatten_columns(summary)


def select_best_rows(
    df: pd.DataFrame,
    group_columns: Iterable[str],
    score_column: str,
    value_columns: Iterable[str],
    *,
    allow_missing_score: bool = False,
    sort_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    group_columns = list(group_columns)
    value_columns = list(value_columns)
    best_rows = []

    for group_values, group_df in df.groupby(group_columns, observed=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row_data = dict(zip(group_columns, group_values))

        valid_df = group_df.dropna(subset=[score_column]) if allow_missing_score else group_df
        if valid_df.empty:
            for column in value_columns:
                row_data[column] = np.nan
        else:
            best_idx = int(valid_df[score_column].idxmax())
            best_row = valid_df.loc[best_idx]
            for column in value_columns:
                row_data[column] = best_row[column]

        best_rows.append(row_data)

    out = pd.DataFrame(best_rows)
    if sort_columns is not None:
        out = out.sort_values(list(sort_columns))
    return out


def safe_mode_int(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")

    modes = clean.mode()
    if modes.empty:
        return float("nan")
    return float(np.min(modes.to_numpy(dtype=float)))
