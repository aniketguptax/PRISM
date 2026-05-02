"""Kalman-ISS recovery sweep on the hierarchical switching Gaussian benchmark."""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prism.continuous.iss import (
    KalmanISSConfig,
    fit_kalman_iss_em,
    one_step_predictive_y,
    solve_steady_state_kalman,
)
from prism.processes.hierarchical_switching_gaussian import HierarchicalSwitchingGaussian
from prism.reconstruction.kalman_iss import (
    _adjusted_rand_index,
    _build_macro_dynamics,
    _normalise_macro_builder,
    _projection_pca,
)
from prism.utils.io import save_csv, save_json


@dataclass(frozen=True)
class SweepSpec:
    noises: tuple[float, ...]
    seeds: tuple[int, ...]
    builders: tuple[str, ...]
    eps_values: tuple[float, ...]
    kmeans_ks: tuple[int, ...]
    length: int
    train_frac: float
    obs_dim: int
    latent_dim: int
    macro_dim: int
    macro_bins: int
    em_iters: int
    history_len: int


def _append_progress(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
        handle.flush()


def _kmeans(
    values: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
    max_iter: int = 100,
) -> np.ndarray:
    n_samples = values.shape[0]
    if n_clusters <= 0:
        raise ValueError("n_clusters must be >= 1.")
    if n_samples == 0:
        return np.zeros((0,), dtype=int)
    if n_clusters == 1:
        return np.zeros(n_samples, dtype=int)

    rng = np.random.default_rng(seed)
    n_clusters = min(n_clusters, n_samples)
    initial = rng.choice(n_samples, size=n_clusters, replace=False)
    centres = values[initial].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(values[:, None, :] - centres[None, :, :], axis=-1)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster in range(n_clusters):
            mask = labels == cluster
            if mask.any():
                centres[cluster] = values[mask].mean(axis=0)
            else:
                centres[cluster] = values[rng.integers(0, n_samples)]
    return labels


def _history_windows(values: np.ndarray, times: np.ndarray, history_len: int) -> np.ndarray:
    times = np.asarray(times, dtype=int)
    starts = times - history_len + 1
    valid = starts >= 0
    if not np.any(valid):
        return np.asarray([], dtype=float)

    valid_starts = starts[valid]
    offsets = np.arange(history_len, dtype=int)
    return np.asarray(
        values[valid_starts[:, None] + offsets].reshape(valid_starts.shape[0], -1),
        dtype=float,
    )


def _gaussian_held_out_nll(
    y_test: np.ndarray,
    mu_pred: np.ndarray,
    cov_pred: np.ndarray,
) -> float:
    if y_test.shape[0] == 0:
        return math.nan
    obs_dim = y_test.shape[1]
    if mu_pred.ndim == 3:
        mu_pred = mu_pred.reshape(mu_pred.shape[0], mu_pred.shape[1])
    total = 0.0
    if cov_pred.shape[0] == y_test.shape[0] and np.all(cov_pred == cov_pred[0]):
        covariance = cov_pred[0]
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return math.nan
        inverse = np.linalg.inv(covariance)
        for idx in range(y_test.shape[0]):
            diff = (y_test[idx] - mu_pred[idx]).reshape(obs_dim, 1)
            quadratic = float((diff.T @ inverse @ diff).item())
            total += 0.5 * (obs_dim * math.log(2.0 * math.pi) + logdet + quadratic)
        return float(total / y_test.shape[0])

    for idx in range(y_test.shape[0]):
        diff = (y_test[idx] - mu_pred[idx]).reshape(obs_dim, 1)
        covariance = cov_pred[idx]
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return math.nan
        inverse = np.linalg.inv(covariance)
        quadratic = float((diff.T @ inverse @ diff).item())
        total += 0.5 * (obs_dim * math.log(2.0 * math.pi) + logdet + quadratic)
    return float(total / y_test.shape[0])


def _ari_row(
    labels: np.ndarray,
    regimes: dict[str, np.ndarray],
    times: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    times = np.asarray(times, dtype=int)
    if labels.shape[0] != times.shape[0]:
        times = times[: labels.shape[0]]
    return {
        "ari_coarse": float(_adjusted_rand_index(labels, regimes["coarse"][times])),
        "ari_fine": float(_adjusted_rand_index(labels, regimes["fine"][times])),
        "ari_joint": float(_adjusted_rand_index(labels, regimes["joint"][times])),
    }


def _plot_summary(rows: list[dict[str, object]], outdir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return
    if not rows:
        return

    df = pd.DataFrame(rows)
    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("ari_coarse", "ARI vs coarse"),
        ("ari_fine", "ARI vs fine"),
        ("ari_joint", "ARI vs joint"),
        ("n_states", "State count"),
        ("unifilarity", "Unifilarity"),
        ("branch_entropy", "Branch entropy"),
    ]
    methods = sorted(df["method"].unique())
    noises = sorted(df["noise"].unique())
    fig, axes = plt.subplots(len(metrics), len(methods), figsize=(3.0 * len(methods), 2.1 * len(metrics)), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for col, method in enumerate(methods):
        sub_method = df[df["method"] == method]
        params = sorted(sub_method["method_param"].unique())
        for row_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx][col]
            for n_idx, noise in enumerate(noises):
                means = []
                for param in params:
                    vals = sub_method[
                        (np.isclose(sub_method["noise"], noise))
                        & (np.isclose(sub_method["method_param"], param))
                    ][metric]
                    vals = vals[np.isfinite(vals)]
                    means.append(float(vals.mean()) if vals.shape[0] else math.nan)
                ax.plot(params, means, marker="o", color=cmap(n_idx / max(len(noises) - 1, 1)), label=f"noise={noise:g}")
            if row_idx == 0:
                ax.set_title(method, fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("eps or k", fontsize=8)
            if metric == "n_states":
                ax.set_yscale("log")
            ax.grid(True, alpha=0.25)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(noises), fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(figures / "hierarchical_switching_recovery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_sweep(spec: SweepSpec, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    progress_path = outdir / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    rows: list[dict[str, object]] = []

    total = len(spec.noises) * len(spec.seeds) * (
        len(spec.builders) * len(spec.eps_values) + 2 * len(spec.kmeans_ks)
    )
    completed = 0
    started = time.perf_counter()
    _append_progress(progress_path, f"sweep start | total_runs={total}")

    for noise in spec.noises:
        for seed in spec.seeds:
            setting_start = time.perf_counter()
            _append_progress(progress_path, f"setting start | noise={noise:g} seed={seed}")
            generator = HierarchicalSwitchingGaussian(
                obs_dim=spec.obs_dim,
                emission_std=float(noise),
            )
            sample = generator.sample(spec.length, seed=seed)
            obs = np.asarray(sample.x, dtype=float)
            latent = np.asarray(sample.latent, dtype=int)
            regimes = generator.regime_labels(latent)
            split = max(2, min(int(spec.length * spec.train_frac), spec.length - 1))
            y_train = obs[:split]
            y_test = obs[split:]

            _append_progress(progress_path, f"em start | noise={noise:g} seed={seed}")
            iss_model = fit_kalman_iss_em(
                y_train,
                KalmanISSConfig(latent_dim=spec.latent_dim, em_iters=spec.em_iters, seed=seed),
            )
            _append_progress(progress_path, f"em done | noise={noise:g} seed={seed}")
            steady_solution = solve_steady_state_kalman(iss_model, strict=False)
            mu_pred_test, cov_pred_test, _ = one_step_predictive_y(
                obs,
                iss_model,
                steady_state=True,
                steady_state_solution=steady_solution,
            )
            gauss_nll = _gaussian_held_out_nll(y_test, mu_pred_test[split:], cov_pred_test[split:])
            projection = _projection_pca(y_train, spec.macro_dim)

            label_payload: dict[str, np.ndarray] = {
                "coarse": regimes["coarse"],
                "fine": regimes["fine"],
                "joint": regimes["joint"],
                "split": np.asarray([split], dtype=int),
            }

            macro_times = np.arange(1, split, dtype=int)
            for builder in spec.builders:
                for eps in spec.eps_values:
                    start = time.perf_counter()
                    _append_progress(
                        progress_path,
                        (
                            f"kalman start | completed={completed}/{total} noise={noise:g} "
                            f"seed={seed} builder={builder} eps={eps:g}"
                        ),
                    )
                    macro = _build_macro_dynamics(
                        y_train=y_train,
                        iss_model=iss_model,
                        projection=projection,
                        eps=float(eps),
                        macro_bins=spec.macro_bins,
                        macro_symboliser="quantile",
                        macro_builder=builder,
                        steady_state=True,
                        steady_state_tol=1e-9,
                        steady_state_max_iter=10_000,
                        steady_state_ridge=1e-9,
                        allow_time_varying_fallback=False,
                        steady_state_solution=steady_solution,
                    )
                    labels = np.asarray(macro.labels, dtype=int)
                    aris = _ari_row(labels, regimes, macro_times)
                    method = f"kalman_{_normalise_macro_builder(builder)}"
                    rows.append(
                        {
                            "noise": float(noise),
                            "seed": int(seed),
                            "method": method,
                            "method_param": float(eps),
                            "n_states": int(macro.n_macro_states),
                            **aris,
                            "gaussian_logloss": float(gauss_nll),
                            "unifilarity": float(macro.unifilarity),
                            "branch_entropy": float(macro.branch_entropy),
                            "elapsed_s": float(time.perf_counter() - start),
                        }
                    )
                    label_payload[f"{method}_eps{eps:g}"] = labels
                    completed += 1
                    _append_progress(
                        progress_path,
                        (
                            f"kalman done | completed={completed}/{total} noise={noise:g} "
                            f"seed={seed} builder={builder} eps={eps:g} states={macro.n_macro_states} "
                            f"ari_joint={aris['ari_joint']:.3f} elapsed={time.perf_counter() - start:.2f}s"
                        ),
                    )

            macro_obs = y_train @ projection.T
            obs_times = np.arange(1, split, dtype=int)
            for k in spec.kmeans_ks:
                start = time.perf_counter()
                labels = _kmeans(macro_obs[1:split], int(k), seed=seed)
                aris = _ari_row(labels, regimes, obs_times)
                rows.append(
                    {
                        "noise": float(noise),
                        "seed": int(seed),
                        "method": "obs_pca_kmeans",
                        "method_param": float(k),
                        "n_states": int(labels.max()) + 1,
                        **aris,
                        "gaussian_logloss": float(gauss_nll),
                        "unifilarity": math.nan,
                        "branch_entropy": math.nan,
                        "elapsed_s": float(time.perf_counter() - start),
                    }
                )
                label_payload[f"obs_pca_kmeans_k{int(k)}"] = labels
                completed += 1
                _append_progress(
                    progress_path,
                    f"obs_pca done | completed={completed}/{total} noise={noise:g} seed={seed} k={k} ari_joint={aris['ari_joint']:.3f}",
                )

            history_times = np.arange(spec.history_len - 1, split, dtype=int)
            history_values = _history_windows(y_train, history_times, spec.history_len)
            for k in spec.kmeans_ks:
                start = time.perf_counter()
                labels = _kmeans(history_values, int(k), seed=seed)
                aris = _ari_row(labels, regimes, history_times)
                rows.append(
                    {
                        "noise": float(noise),
                        "seed": int(seed),
                        "method": "history_kmeans",
                        "method_param": float(k),
                        "n_states": int(labels.max()) + 1,
                        **aris,
                        "gaussian_logloss": float(gauss_nll),
                        "unifilarity": math.nan,
                        "branch_entropy": math.nan,
                        "elapsed_s": float(time.perf_counter() - start),
                    }
                )
                label_payload[f"history_kmeans_k{int(k)}"] = labels
                completed += 1
                _append_progress(
                    progress_path,
                    f"history done | completed={completed}/{total} noise={noise:g} seed={seed} k={k} ari_joint={aris['ari_joint']:.3f}",
                )

            np.savez(outdir / f"labels_noise{noise:g}_seed{seed}.npz", **label_payload)
            _append_progress(
                progress_path,
                f"setting done | noise={noise:g} seed={seed} elapsed={time.perf_counter() - setting_start:.2f}s",
            )

    fieldnames = [
        "noise",
        "seed",
        "method",
        "method_param",
        "n_states",
        "ari_coarse",
        "ari_fine",
        "ari_joint",
        "gaussian_logloss",
        "unifilarity",
        "branch_entropy",
        "elapsed_s",
    ]
    save_csv(outdir / "recovery.csv", rows, append=False, fieldnames=fieldnames)
    save_json(
        outdir / "sweep_spec.json",
        {
            "noises": list(spec.noises),
            "seeds": list(spec.seeds),
            "builders": list(spec.builders),
            "eps_values": list(spec.eps_values),
            "kmeans_ks": list(spec.kmeans_ks),
            "length": spec.length,
            "train_frac": spec.train_frac,
            "obs_dim": spec.obs_dim,
            "latent_dim": spec.latent_dim,
            "macro_dim": spec.macro_dim,
            "macro_bins": spec.macro_bins,
            "em_iters": spec.em_iters,
            "history_len": spec.history_len,
        },
    )
    _plot_summary(rows, outdir)
    _append_progress(
        progress_path,
        f"sweep complete | rows={len(rows)} csv=recovery.csv elapsed={time.perf_counter() - started:.1f}s",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noises", nargs="+", type=float, default=[0.10, 0.20, 0.35])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--builders", nargs="+", default=["hierarchical_single", "linear_quantile"])
    parser.add_argument("--eps-values", nargs="+", type=float, default=[0.15, 0.25, 0.35, 0.50])
    parser.add_argument("--kmeans-ks", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--macro-dim", type=int, default=2)
    parser.add_argument("--macro-bins", type=int, default=3)
    parser.add_argument("--em-iters", type=int, default=30)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path("./results/hierarchical_switching_sweep"))
    args = parser.parse_args()

    spec = SweepSpec(
        noises=tuple(args.noises),
        seeds=tuple(args.seeds),
        builders=tuple(args.builders),
        eps_values=tuple(args.eps_values),
        kmeans_ks=tuple(args.kmeans_ks),
        length=int(args.length),
        train_frac=float(args.train_frac),
        obs_dim=int(args.obs_dim),
        latent_dim=int(args.latent_dim),
        macro_dim=int(args.macro_dim),
        macro_bins=int(args.macro_bins),
        em_iters=int(args.em_iters),
        history_len=int(args.history_len),
    )
    run_sweep(spec, args.outdir)


if __name__ == "__main__":
    main()
