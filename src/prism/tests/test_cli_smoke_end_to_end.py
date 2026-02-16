import csv
import math
import os
import subprocess
import sys
from pathlib import Path


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


def _is_finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def test_discrete_cli_end_to_end_iid_markov_even(tmp_path: Path) -> None:
    processes = ("iid_bernoulli", "markov_order_1", "even_process")
    for process in processes:
        outdir = tmp_path / f"{process}_run"
        _run_module(
            "prism.cli",
            [
                "--process",
                process,
                "--reconstructor",
                "one_step",
                "--ks",
                "1",
                "2",
                "--length",
                "1200",
                "--train-frac",
                "0.8",
                "--seeds",
                "0",
                "--outdir",
                str(outdir),
                "--force",
            ],
            tmp_path,
        )
        assert (outdir / "runs.csv").exists()

        _run_module("prism.analysis.summarise", ["--root", str(outdir)], tmp_path)
        assert (outdir / "summary_by_condition.csv").exists()


def test_continuous_cli_sweep_d_and_dv(tmp_path: Path) -> None:
    outdir = tmp_path / "continuous_run"
    _run_module(
        "prism.cli",
        [
            "--process",
            "linear_gaussian_ssm",
            "--reconstructor",
            "kalman_iss",
            "--ks",
            "1",
            "2",
            "--dvs",
            "1",
            "2",
            "--em-iters",
            "20",
            "--macro-eps",
            "0.25",
            "--macro-bins",
            "3",
            "--length",
            "1200",
            "--train-frac",
            "0.8",
            "--seeds",
            "0",
            "--outdir",
            str(outdir),
            "--force",
        ],
        tmp_path,
    )
    runs_csv = outdir / "runs.csv"
    assert runs_csv.exists()

    with runs_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4

    for row in rows:
        for metric in (
            "gaussian_logloss",
            "C_mu_empirical",
            "unifilarity_score",
            "branch_entropy",
            "n_states",
            "macro_dim",
            "obs_dim",
            "latent_dim",
        ):
            assert _is_finite(row[metric]), f"{metric} must be finite, got {row[metric]!r}"

    loss_by_pair = {(int(float(r["k"])), int(float(r["dv"]))): float(r["gaussian_logloss"]) for r in rows}
    assert loss_by_pair[(2, 1)] <= loss_by_pair[(1, 1)] + 0.2

    _run_module("prism.analysis.summarise", ["--root", str(outdir)], tmp_path)
    assert (outdir / "summary_by_condition.csv").exists()


def test_cli_reports_progress_logs(tmp_path: Path) -> None:
    outdir = tmp_path / "progress_run"
    proc = _run_module(
        "prism.cli",
        [
            "--process",
            "iid_bernoulli",
            "--reconstructor",
            "one_step",
            "--ks",
            "1",
            "--length",
            "800",
            "--train-frac",
            "0.8",
            "--seeds",
            "0",
            "--outdir",
            str(outdir),
            "--force",
            "--log-level",
            "INFO",
        ],
        tmp_path,
    )

    # logging.basicConfig writes to stderr by default
    assert "Running PRISM |" in proc.stderr
    assert "Condition 1/1" in proc.stderr
    assert "Progress 1/1" in proc.stderr
