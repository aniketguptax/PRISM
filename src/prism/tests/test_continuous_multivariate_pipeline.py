import csv
import math
from pathlib import Path

from prism.experiments.runner import run_experiment
from prism.processes.linear_gaussian_ssm import LinearGaussianSSM
from prism.reconstruction.kalman_iss import KalmanISSReconstructor
from prism.representations.continuous import ISSDim


def _is_finite(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except (TypeError, ValueError):
        return False


def test_continuous_runner_sweeps_d_and_dv(tmp_path: Path) -> None:
    process = LinearGaussianSSM(latent_dim=2, obs_dim=3, a=0.9, process_std=0.25, obs_std=0.2)
    reconstructor = KalmanISSReconstructor(
        em_iters=20,
        em_tol=1e-4,
        em_ridge=1e-6,
        macro_eps=0.25,
        macro_bins=3,
        projection_mode="pca",
    )
    reps = [ISSDim(d=1, dv=1), ISSDim(d=2, dv=2)]

    run_experiment(
        process=process,
        reconstructor=reconstructor,
        representations=reps,
        length=900,
        train_frac=0.8,
        seeds=[0],
        outdir=tmp_path,
    )

    runs_csv = tmp_path / "runs.csv"
    assert runs_csv.exists()
    with runs_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2

    for row in rows:
        assert row["process"] == "linear_gaussian_ssm"
        assert row["reconstructor"] == "kalman_iss"
        assert _is_finite(row["logloss"])
        assert _is_finite(row["gaussian_logloss"])
        assert _is_finite(row["n_states"])
        assert _is_finite(row["C_mu_empirical"])
        assert _is_finite(row["unifilarity_score"])
        assert _is_finite(row["branch_entropy"])
        assert _is_finite(row["psi_opt"])
        assert _is_finite(row["macro_dim"])
        assert _is_finite(row["obs_dim"])
        assert _is_finite(row["latent_dim"])
        assert row["projection_mode"] in {"pca", "random", "psi_opt"}


def test_continuous_loss_not_worse_at_true_latent_dim(tmp_path: Path) -> None:
    process = LinearGaussianSSM(latent_dim=2, obs_dim=3, a=0.88, process_std=0.22, obs_std=0.18)
    reconstructor = KalmanISSReconstructor(
        em_iters=25,
        em_tol=1e-4,
        em_ridge=1e-6,
        macro_eps=0.25,
        macro_bins=3,
        projection_mode="pca",
    )
    reps = [ISSDim(d=1, dv=1), ISSDim(d=2, dv=1)]

    run_experiment(
        process=process,
        reconstructor=reconstructor,
        representations=reps,
        length=1200,
        train_frac=0.8,
        seeds=[0],
        outdir=tmp_path,
    )

    with (tmp_path / "runs.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    losses = {int(float(row["k"])): float(row["gaussian_logloss"]) for row in rows}
    assert 1 in losses and 2 in losses
    # EM on finite data is noisy; require "not materially worse" at the true latent order.
    assert losses[2] <= losses[1] + 0.2
