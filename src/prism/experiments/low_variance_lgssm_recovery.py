"""Kalman predictive-state recovery for the low-variance LGSSM benchmark."""

from __future__ import annotations

import argparse
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
from prism.processes.predictive_low_variance_lgssm import PredictiveLowVarianceLGSSM
from prism.reconstruction.kalman_iss import _adjusted_rand_index, _projection_pca
from prism.utils.io import save_csv, save_json


@dataclass(frozen=True)
class SweepSpec:
    obs_stds: tuple[float, ...]
    seeds: tuple[int, ...]
    eps_values: tuple[float, ...]
    kmeans_ks: tuple[int, ...]
    length: int
    train_frac: float
    obs_dim: int
    latent_dim: int
    em_iters: int
    history_len: int


def _append_progress(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
        handle.flush()


def _kmeans(values: np.ndarray, n_clusters: int, *, seed: int, max_iter: int = 100) -> np.ndarray:
    n_samples = values.shape[0]
    if n_clusters <= 1:
        return np.zeros(n_samples, dtype=int)
    rng = np.random.default_rng(seed)
    n_clusters = min(n_clusters, n_samples)
    centres = values[rng.choice(n_samples, size=n_clusters, replace=False)].copy()
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(values[:, None, :] - centres[None, :, :], axis=-1)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster in range(n_clusters):
            mask = labels == cluster
            centres[cluster] = values[mask].mean(axis=0) if mask.any() else values[rng.integers(0, n_samples)]
    return labels


def _single_link(values: np.ndarray, eps: float) -> np.ndarray:
    n = values.shape[0]
    labels = np.full(n, -1, dtype=int)
    cluster = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        labels[start] = cluster
        stack = [start]
        while stack:
            idx = stack.pop()
            distances = np.linalg.norm(values - values[idx], axis=1)
            neighbours = np.where((distances <= eps) & (labels < 0))[0]
            for neighbour in neighbours.tolist():
                labels[neighbour] = cluster
                stack.append(int(neighbour))
        cluster += 1
    return labels


def _history_windows(values: np.ndarray, times: np.ndarray, history_len: int) -> np.ndarray:
    times = np.asarray(times, dtype=int)
    starts = times - history_len + 1
    if times.size == 0 or np.any(starts < 0):
        return np.asarray(
            [values[t - history_len + 1 : t + 1].reshape(-1) for t in times.tolist()],
            dtype=float,
        )

    offsets = np.arange(history_len, dtype=int)
    return np.asarray(values[starts[:, None] + offsets].reshape(times.shape[0], -1), dtype=float)


def _ari_row(labels: np.ndarray, regimes: dict[str, np.ndarray], times: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    times = np.asarray(times[: labels.shape[0]], dtype=int)
    return {
        "ari_slow": float(_adjusted_rand_index(labels, regimes["slow"][times])),
        "ari_fast": float(_adjusted_rand_index(labels, regimes["fast"][times])),
        "ari_joint": float(_adjusted_rand_index(labels, regimes["joint"][times])),
    }


def _gaussian_nll(y: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    if mu.ndim == 3:
        mu = mu.reshape(mu.shape[0], mu.shape[1])
    total = 0.0
    p = y.shape[1]
    if cov.shape[0] == y.shape[0] and np.all(cov == cov[0]):
        covariance = cov[0]
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return math.nan
        inverse = np.linalg.inv(covariance)
        for idx in range(y.shape[0]):
            diff = (y[idx] - mu[idx]).reshape(p, 1)
            quadratic = float((diff.T @ inverse @ diff).item())
            total += 0.5 * (p * math.log(2.0 * math.pi) + logdet + quadratic)
        return float(total / max(y.shape[0], 1))

    for idx in range(y.shape[0]):
        diff = (y[idx] - mu[idx]).reshape(p, 1)
        sign, logdet = np.linalg.slogdet(cov[idx])
        if sign <= 0:
            return math.nan
        quadratic = float((diff.T @ np.linalg.inv(cov[idx]) @ diff).item())
        total += 0.5 * (p * math.log(2.0 * math.pi) + logdet + quadratic)
    return float(total / max(y.shape[0], 1))


def run_sweep(spec: SweepSpec, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    progress = outdir / "progress.log"
    progress.write_text("", encoding="utf-8")
    rows: list[dict[str, object]] = []
    total = len(spec.obs_stds) * len(spec.seeds) * (
        len(spec.eps_values) + 3 * len(spec.kmeans_ks)
    )
    completed = 0
    started = time.perf_counter()
    _append_progress(progress, f"sweep start | total_runs={total}")

    for obs_std in spec.obs_stds:
        for seed in spec.seeds:
            setting_start = time.perf_counter()
            _append_progress(progress, f"setting start | obs_std={obs_std:g} seed={seed}")
            process = PredictiveLowVarianceLGSSM(
                obs_std=float(obs_std),
                obs_dim=spec.obs_dim,
                latent_dim=spec.latent_dim,
            )
            sample = process.sample(spec.length, seed=seed)
            obs = np.asarray(sample.x, dtype=float)
            latent = np.asarray(sample.latent, dtype=float)
            regimes = process.regime_labels(latent)
            split = int(spec.length * spec.train_frac)
            y_train = obs[:split]
            y_all = obs

            _append_progress(progress, f"em start | obs_std={obs_std:g} seed={seed}")
            model = fit_kalman_iss_em(
                y_train,
                KalmanISSConfig(latent_dim=spec.latent_dim, em_iters=spec.em_iters, seed=seed),
            )
            steady = solve_steady_state_kalman(model, strict=False)
            mu_y, cov_y, _ = one_step_predictive_y(
                y_all,
                model,
                steady_state=True,
                steady_state_solution=steady,
            )
            _append_progress(progress, f"em done | obs_std={obs_std:g} seed={seed}")
            nll = _gaussian_nll(obs[split:], mu_y[split:], cov_y[split:])

            times = np.arange(1, split, dtype=int)
            pred_values = mu_y.reshape(mu_y.shape[0], mu_y.shape[1])[times]
            pred_values = (pred_values - pred_values.mean(axis=0)) / np.maximum(pred_values.std(axis=0), 1e-9)
            for eps in spec.eps_values:
                start = time.perf_counter()
                labels = _single_link(pred_values, float(eps))
                aris = _ari_row(labels, regimes, times)
                rows.append(
                    {
                        "obs_std": float(obs_std),
                        "seed": int(seed),
                        "method": "kalman_predictive_single",
                        "method_param": float(eps),
                        "n_states": int(labels.max()) + 1,
                        **aris,
                        "gaussian_logloss": nll,
                        "elapsed_s": float(time.perf_counter() - start),
                    }
                )
                completed += 1
                _append_progress(progress, f"kalman done | completed={completed}/{total} eps={eps:g} states={int(labels.max()) + 1} ari_slow={aris['ari_slow']:.3f}")

            for k in spec.kmeans_ks:
                start = time.perf_counter()
                labels = _kmeans(pred_values, int(k), seed=seed)
                aris = _ari_row(labels, regimes, times)
                rows.append(
                    {
                        "obs_std": float(obs_std),
                        "seed": int(seed),
                        "method": "kalman_predictive_kmeans",
                        "method_param": float(k),
                        "n_states": int(labels.max()) + 1,
                        **aris,
                        "gaussian_logloss": nll,
                        "elapsed_s": float(time.perf_counter() - start),
                    }
                )
                completed += 1
                _append_progress(progress, f"kalman-kmeans done | completed={completed}/{total} k={k} ari_slow={aris['ari_slow']:.3f}")

            projection = _projection_pca(y_train, min(2, y_train.shape[1]))
            obs_pca = y_train @ projection.T
            for k in spec.kmeans_ks:
                start = time.perf_counter()
                labels = _kmeans(obs_pca[times], int(k), seed=seed)
                aris = _ari_row(labels, regimes, times)
                rows.append(
                    {
                        "obs_std": float(obs_std),
                        "seed": int(seed),
                        "method": "obs_pca_kmeans",
                        "method_param": float(k),
                        "n_states": int(labels.max()) + 1,
                        **aris,
                        "gaussian_logloss": nll,
                        "elapsed_s": float(time.perf_counter() - start),
                    }
                )
                completed += 1
                _append_progress(progress, f"obs done | completed={completed}/{total} k={k} ari_slow={aris['ari_slow']:.3f}")

            history_times = np.arange(spec.history_len - 1, split, dtype=int)
            history = _history_windows(y_train, history_times, spec.history_len)
            for k in spec.kmeans_ks:
                start = time.perf_counter()
                labels = _kmeans(history, int(k), seed=seed)
                aris = _ari_row(labels, regimes, history_times)
                rows.append(
                    {
                        "obs_std": float(obs_std),
                        "seed": int(seed),
                        "method": "history_kmeans",
                        "method_param": float(k),
                        "n_states": int(labels.max()) + 1,
                        **aris,
                        "gaussian_logloss": nll,
                        "elapsed_s": float(time.perf_counter() - start),
                    }
                )
                completed += 1
                _append_progress(progress, f"history done | completed={completed}/{total} k={k} ari_slow={aris['ari_slow']:.3f}")

            _append_progress(progress, f"setting done | obs_std={obs_std:g} seed={seed} elapsed={time.perf_counter() - setting_start:.2f}s")

    fieldnames = [
        "obs_std",
        "seed",
        "method",
        "method_param",
        "n_states",
        "ari_slow",
        "ari_fast",
        "ari_joint",
        "gaussian_logloss",
        "elapsed_s",
    ]
    save_csv(outdir / "recovery.csv", rows, append=False, fieldnames=fieldnames)
    save_json(
        outdir / "sweep_spec.json",
        {
            "obs_stds": list(spec.obs_stds),
            "seeds": list(spec.seeds),
            "eps_values": list(spec.eps_values),
            "kmeans_ks": list(spec.kmeans_ks),
            "length": spec.length,
            "train_frac": spec.train_frac,
            "obs_dim": spec.obs_dim,
            "latent_dim": spec.latent_dim,
            "em_iters": spec.em_iters,
            "history_len": spec.history_len,
        },
    )
    _append_progress(progress, f"sweep complete | rows={len(rows)} elapsed={time.perf_counter() - started:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-stds", nargs="+", type=float, default=[0.15, 0.25, 0.40])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eps-values", nargs="+", type=float, default=[0.2, 0.35, 0.5, 0.8, 1.2])
    parser.add_argument("--kmeans-ks", nargs="+", type=int, default=[3, 6, 9])
    parser.add_argument("--length", type=int, default=6000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--em-iters", type=int, default=30)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path("./results/low_variance_lgssm_sweep"))
    args = parser.parse_args()
    spec = SweepSpec(
        obs_stds=tuple(args.obs_stds),
        seeds=tuple(args.seeds),
        eps_values=tuple(args.eps_values),
        kmeans_ks=tuple(args.kmeans_ks),
        length=int(args.length),
        train_frac=float(args.train_frac),
        obs_dim=int(args.obs_dim),
        latent_dim=int(args.latent_dim),
        em_iters=int(args.em_iters),
        history_len=int(args.history_len),
    )
    run_sweep(spec, args.outdir)


if __name__ == "__main__":
    main()
