from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .protocols import Process, Sample

def load_1d_timeseries(path: Path, column: Optional[int] = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            c = 0 if column is None else int(column)
            return arr[:, c]
        raise ValueError(f"Unsupported npy shape: {arr.shape}")

    if path.suffix == ".csv":
        import csv
        col = 0 if column is None else int(column)
        xs = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    xs.append(float(row[col]))
                except (ValueError, IndexError):
                    continue
        return np.asarray(xs, dtype=float)

    # whitespace text
    txt = path.read_text(encoding="utf-8").strip().split()
    return np.asarray([float(v) for v in txt], dtype=float)

@dataclass(frozen=True)
class ContinuousFile(Process):
    path: Path
    column: Optional[int] = None

    @property
    def name(self) -> str:
        return f"continuous_{self.path.stem}"

    def sample(self, length: int, seed: int) -> Sample:
        y = load_1d_timeseries(self.path, self.column)
        y = y[: int(length)]
        return Sample(x=[float(v) for v in y.tolist()], latent=None)