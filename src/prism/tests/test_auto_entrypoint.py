import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[2]


def _module_python() -> str:
    override = os.environ.get("PRISM_TEST_PYTHON")
    if override:
        return override
    return sys.executable


def _run_module(module: str, args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(SRC_ROOT),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "mplconfig"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
    }
    for key in ("TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL"):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return subprocess.run(
        [_module_python(), "-m", module, *args],
        cwd=SRC_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_series(path: Path, rows: int = 260, cols: int = 3) -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(rows, cols))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(data.tolist())


def _auto_args(data_path: Path, outdir: Path) -> list[str]:
    return [
        str(data_path),
        "--outdir",
        str(outdir),
        "--force",
        "--length",
        "220",
        "--train-frac",
        "0.8",
        "--seeds",
        "0",
        "--ks",
        "1",
        "--dvs",
        "1",
        "--selection-projections",
        "pca",
        "random",
        "--selection-builders",
        "greedy",
        "hierarchical_single",
        "--selection-eps",
        "0.2",
        "0.25",
        "--selection-repeats",
        "2",
        "--em-iters",
        "8",
        "--psi-restarts",
        "1",
        "--psi-iters",
        "8",
    ]


def test_auto_module_runs_and_writes_outputs(tmp_path: Path) -> None:
    data_path = tmp_path / "series.csv"
    outdir = tmp_path / "auto_run"
    _write_series(data_path)
    _run_module("prism.auto", _auto_args(data_path, outdir), tmp_path)

    assert (outdir / "runs.csv").exists()
    assert (outdir / "summary_by_condition.csv").exists()
    assert (outdir / "best_selection.json").exists()
    assert (outdir / "best_pipeline" / "runs.csv").exists()
    assert (outdir / "best_pipeline" / "summary_by_condition.csv").exists()
    assert (outdir / "tractability_by_builder.csv").exists()


def test_prism_module_dispatches_to_auto_for_data_path(tmp_path: Path) -> None:
    data_path = tmp_path / "series.csv"
    outdir = tmp_path / "dispatch_run"
    _write_series(data_path)
    _run_module("prism", _auto_args(data_path, outdir), tmp_path)
    assert (outdir / "best_selection.json").exists()
