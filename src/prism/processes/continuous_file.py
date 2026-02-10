"""Continuous process that reads observations from file."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .protocols import Process, Sample


def load_timeseries(path: Path, column: Optional[int] = None) -> np.ndarray:
    """Load a scalar continuous time series from .npy/.csv/.txt."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=float)
        if arr.ndim == 1:
            data = arr
        elif arr.ndim == 2:
            if column is None:
                raise ValueError(
                    f"{path} is 2D with shape {arr.shape}; pass --data-column to select a single channel."
                )
            if column < 0 or column >= arr.shape[1]:
                raise IndexError(f"--data-column must be in [0, {arr.shape[1] - 1}], got {column}.")
            data = arr[:, column]
        else:
            raise ValueError(f"Unsupported npy shape {arr.shape}.")
    elif path.suffix == ".csv":
        import csv

        target_col = 0 if column is None else int(column)
        values: list[float] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row_idx, row in enumerate(csv.reader(handle)):
                if not row:
                    continue
                if target_col >= len(row):
                    raise IndexError(
                        f"Row {row_idx} in {path} has {len(row)} columns; --data-column={target_col} is invalid."
                    )
                try:
                    values.append(float(row[target_col]))
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value {row[target_col]!r} at row {row_idx}, column {target_col} in {path}."
                    ) from exc
        data = np.asarray(values, dtype=float)
    else:
        tokens = path.read_text(encoding="utf-8").strip().split()
        data = np.asarray([float(token) for token in tokens], dtype=float)

    if data.size == 0:
        raise ValueError(f"No observations found in {path}.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path} contains non-finite values; clean the dataset before running PRISM.")
    return data


@dataclass(frozen=True)
class ContinuousFile(Process):
    """Expose a file-backed scalar time series as a PRISM process."""

    path: Path
    column: Optional[int] = None

    @property
    def name(self) -> str:
        return f"continuous_{self.path.stem}"

    def sample(self, length: int, seed: int) -> Sample:
        del seed  # File-backed process is deterministic once loaded.
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}.")
        data = load_timeseries(self.path, self.column)
        truncated = data[: int(length)]
        return Sample(x=[float(v) for v in truncated.tolist()], latent=None)
