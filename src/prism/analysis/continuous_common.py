import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def safe_int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def filter_rows(
    rows: Sequence[Dict[str, str]],
    *,
    base_process: Optional[str],
    condition_id: Optional[str],
    subsample_step: Optional[int],
    flip_p: Optional[float],
    projection_modes: Optional[Sequence[str]],
) -> List[Dict[str, str]]:
    mode_filter = None if projection_modes is None else {mode.strip() for mode in projection_modes if mode.strip()}
    filtered: List[Dict[str, str]] = []
    for row in rows:
        if base_process is not None and row.get("base_process", "") != base_process:
            continue
        if condition_id is not None and row.get("condition_id", "") != condition_id:
            continue
        if subsample_step is not None:
            row_step = safe_int(row.get("subsample_step", ""))
            if row_step is None or row_step != subsample_step:
                continue
        if flip_p is not None:
            row_flip = safe_float(row.get("flip_p", ""))
            if row_flip is None or abs(row_flip - flip_p) > 1e-12:
                continue
        if mode_filter is not None:
            mode = row.get("projection_mode", "").strip()
            if mode not in mode_filter:
                continue
        filtered.append(row)
    return filtered


def available_projection_modes(rows: Sequence[Dict[str, str]]) -> List[str]:
    discovered = sorted({row.get("projection_mode", "").strip() for row in rows if row.get("projection_mode", "").strip()})
    if discovered:
        return discovered
    return ["unknown"]


def metric_column(rows: Sequence[Dict[str, str]], metric: str) -> str:
    if not rows:
        raise ValueError("No rows available to resolve metric columns.")
    columns = set(rows[0].keys())
    if metric in columns:
        return metric
    mean_col = f"{metric}_mean"
    if mean_col in columns:
        return mean_col
    raise ValueError(f"Metric {metric!r} not found. Available columns include: {sorted(columns)}")


def build_axes(rows: Sequence[Dict[str, str]]) -> Tuple[List[int], List[int]]:
    ks = sorted(
        k for k in {safe_int(row.get("k", "")) for row in rows} if k is not None
    )
    dvs = sorted(
        dv for dv in {safe_int(row.get("dv", "")) for row in rows} if dv is not None
    )
    return ks, dvs


def mode_matrix(
    rows: Sequence[Dict[str, str]],
    *,
    projection_mode: str,
    metric_col: str,
    ks: Sequence[int],
    dvs: Sequence[int],
) -> np.ndarray:
    matrix = np.full((len(dvs), len(ks)), np.nan, dtype=float)
    if not ks or not dvs:
        return matrix
    k_to_i = {k: idx for idx, k in enumerate(ks)}
    dv_to_j = {dv: idx for idx, dv in enumerate(dvs)}
    for row in rows:
        row_mode = row.get("projection_mode", "").strip() or "unknown"
        if row_mode != projection_mode:
            continue
        k = safe_int(row.get("k", ""))
        dv = safe_int(row.get("dv", ""))
        if k is None or dv is None:
            continue
        if k not in k_to_i or dv not in dv_to_j:
            continue
        value = safe_float(row.get(metric_col, ""))
        if value is None:
            continue
        matrix[dv_to_j[dv], k_to_i[k]] = value
    return matrix


def finite_min_max(matrices: Iterable[np.ndarray]) -> Tuple[Optional[float], Optional[float]]:
    values: List[float] = []
    for matrix in matrices:
        finite = matrix[np.isfinite(matrix)]
        if finite.size:
            values.extend(float(v) for v in finite.tolist())
    if not values:
        return None, None
    return min(values), max(values)
