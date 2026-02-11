import csv
import os
import subprocess
import sys
from pathlib import Path

from prism.analysis.summarise import summarise


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


def test_summarise_keeps_projection_modes_separate(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.csv"
    rows = [
        {
            "seed": "0",
            "base_process": "linear_gaussian_ssm",
            "condition_id": "flip0_sub1",
            "flip_p": "0.0",
            "subsample_step": "1",
            "reconstructor": "kalman_iss",
            "projection_mode": "pca",
            "representation": "iss_d4_dv1",
            "k": "4",
            "dv": "1",
            "logloss": "1.25",
            "gaussian_logloss": "1.25",
            "n_states": "12",
        },
        {
            "seed": "1",
            "base_process": "linear_gaussian_ssm",
            "condition_id": "flip0_sub1",
            "flip_p": "0.0",
            "subsample_step": "1",
            "reconstructor": "kalman_iss",
            "projection_mode": "random",
            "representation": "iss_d4_dv1",
            "k": "4",
            "dv": "1",
            "logloss": "2.75",
            "gaussian_logloss": "2.75",
            "n_states": "14",
        },
    ]
    with runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarise(tmp_path)

    with (tmp_path / "summary_by_condition.csv").open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 2
    by_mode = {row["projection_mode"]: row for row in summary_rows}
    assert float(by_mode["pca"]["logloss_mean"]) == 1.25
    assert float(by_mode["random"]["logloss_mean"]) == 2.75

    with (tmp_path / "summary_simple.csv").open("r", encoding="utf-8", newline="") as handle:
        simple_header = next(csv.reader(handle))
    assert "projection_mode" in simple_header


def _write_summary_csv(path: Path) -> None:
    rows = []
    for mode, base in [("pca", 1.0), ("random", 1.4), ("psi_opt", 0.9)]:
        for k in (2, 4):
            for dv in (1, 2):
                rows.append(
                    {
                        "base_process": "linear_gaussian_ssm",
                        "condition_id": "flip0_sub1",
                        "flip_p": "0.0",
                        "subsample_step": "1",
                        "reconstructor": "kalman_iss",
                        "projection_mode": mode,
                        "representation": f"iss_d{k}_dv{dv}",
                        "k": str(k),
                        "dv": str(dv),
                        "n": "3",
                        "gaussian_logloss_mean": str(base + 0.1 * k + 0.05 * dv),
                        "gaussian_logloss_std": "0.01",
                        "n_states_mean": str(20 + 2 * k + dv),
                        "n_states_std": "0.2",
                        "C_mu_empirical_mean": str(2.0 + 0.2 * k + 0.05 * dv),
                        "C_mu_empirical_std": "0.03",
                        "psi_opt_mean": str(0.5 + 0.01 * k - 0.02 * dv),
                        "psi_opt_std": "0.01",
                        "unifilarity_score_mean": "0.98",
                        "unifilarity_score_std": "0.01",
                        "branch_entropy_mean": "0.02",
                        "branch_entropy_std": "0.01",
                    }
                )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_continuous_heatmaps_writes_projection_files(tmp_path: Path) -> None:
    root = tmp_path / "cont"
    root.mkdir()
    summary = root / "summary_by_condition.csv"
    _write_summary_csv(summary)

    _run_module(
        "prism.analysis.continuous_heatmaps",
        [
            "--root",
            str(root),
            "--metrics",
            "gaussian_logloss",
            "n_states",
            "--shared-scale",
        ],
        tmp_path,
    )

    figures = root / "figures" / "continuous_heatmaps"
    assert (figures / "pca_gaussian_logloss_mean_heatmap.png").exists()
    assert (figures / "random_n_states_mean_heatmap.png").exists()
    assert (figures / "psi_opt_gaussian_logloss_mean_heatmap.png").exists()


def test_compare_projection_modes_writes_side_by_side_plot(tmp_path: Path) -> None:
    root = tmp_path / "cont"
    root.mkdir()
    summary = root / "summary_by_condition.csv"
    _write_summary_csv(summary)

    _run_module(
        "prism.analysis.compare_projection_modes",
        [
            "--root",
            str(root),
            "--metrics",
            "gaussian_logloss",
        ],
        tmp_path,
    )

    compare_plot = root / "figures" / "projection_mode_comparison" / "compare_projection_modes_gaussian_logloss_mean.png"
    assert compare_plot.exists()
