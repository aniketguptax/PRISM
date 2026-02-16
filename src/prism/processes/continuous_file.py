from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from prism.types import Obs

from .protocols import Process, Sample


def _validate_columns(n_cols: int, columns: Optional[Sequence[int]]) -> tuple[int, ...]:
    if columns is None:
        return tuple(range(n_cols))
    if len(columns) == 0:
        raise ValueError("columns must contain at least one index.")
    selected: list[int] = []
    for column in columns:
        idx = int(column)
        if idx < 0 or idx >= n_cols:
            raise IndexError(f"Column index {idx} is out of range [0, {n_cols - 1}].")
        selected.append(idx)
    return tuple(selected)


def _rows_to_obs(rows: np.ndarray) -> list[Obs]:
    if rows.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {rows.shape}.")
    if rows.shape[1] == 1:
        return [float(value) for value in rows[:, 0]]
    return [tuple(float(v) for v in row) for row in rows]


def load_timeseries(
    path: Path,
    columns: Optional[Sequence[int]] = None,
    column: Optional[int] = None,
) -> np.ndarray:
    """
    Load a continuous time series matrix with shape ``(T, p)``.

    .npy accepts 1D or 2D arrays.
    .csv accepts dense numeric rows.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=float)
        if arr.ndim == 1:
            matrix = arr.reshape(-1, 1)
        elif arr.ndim == 2:
            matrix = arr
        else:
            raise ValueError(f"Unsupported npy shape {arr.shape}. Expected 1D or 2D.")
    elif path.suffix == ".csv":
        import csv

        csv_rows: list[list[float]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row_idx, row in enumerate(csv.reader(handle)):
                if not row:
                    continue
                values: list[float] = []
                for col_idx, cell in enumerate(row):
                    try:
                        values.append(float(cell))
                    except ValueError as exc:
                        raise ValueError(
                            f"Non-numeric value {cell!r} at row {row_idx}, column {col_idx} in {path}."
                        ) from exc
                csv_rows.append(values)
        if not csv_rows:
            raise ValueError(f"No observations found in {path}.")
        width = len(csv_rows[0])
        if width == 0:
            raise ValueError(f"{path} contains empty rows.")
        for row_idx, row in enumerate(csv_rows[1:], start=1):
            if len(row) != width:
                raise ValueError(
                    f"Row {row_idx} in {path} has width {len(row)}; expected {width}."
                )
        matrix = np.asarray(csv_rows, dtype=float)
    else:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"No observations found in {path}.")
        text_rows = [line.split() for line in text.splitlines() if line.strip()]
        width = len(text_rows[0])
        parsed: list[list[float]] = []
        for row_idx, row in enumerate(text_rows):
            if len(row) != width:
                raise ValueError(
                    f"Row {row_idx} in {path} has width {len(row)}; expected {width}."
                )
            try:
                parsed.append([float(token) for token in row])
            except ValueError as exc:
                raise ValueError(f"Non-numeric token in row {row_idx} of {path}.") from exc
        matrix = np.asarray(parsed, dtype=float)

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"No observations found in {path}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path} contains non-finite values; clean the dataset before running PRISM.")

    if column is not None and columns is not None:
        raise ValueError("Use either column or columns, not both.")
    selected_columns = columns if columns is not None else ((column,) if column is not None else None)
    selected = _validate_columns(matrix.shape[1], selected_columns)
    matrix = matrix[:, selected]
    return matrix


@dataclass(frozen=True)
class ContinuousFile(Process):
    """Expose file-backed multivariate continuous observations as a PRISM process"""

    path: Path
    column: Optional[int] = None
    columns: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.column is not None and self.columns is not None:
            raise ValueError("Use either column or columns, not both.")

    @property
    def selected_columns(self) -> Optional[tuple[int, ...]]:
        if self.columns is not None:
            return self.columns
        if self.column is not None:
            return (self.column,)
        return None

    @property
    def name(self) -> str:
        return f"continuous_{self.path.stem}"

    def sample(self, length: int, seed: int) -> Sample:
        del seed  # deterministic once loaded from file
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")
        matrix = load_timeseries(self.path, columns=self.selected_columns)
        if matrix.shape[0] < int(length):
            raise ValueError(
                f"Requested length={length}, but {self.path} only has {matrix.shape[0]} rows."
            )
        clipped = matrix[: int(length)]
        return Sample(x=_rows_to_obs(clipped), latent=None)
