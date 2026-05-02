"""Run the block-modular recovery sweep."""

from __future__ import annotations

import argparse
import logging
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
from prism.processes.block_modular_lgssm import BlockModularLGSSM
from prism.reconstruction.kalman_iss import (
    _adjusted_rand_index,
    _build_macro_dynamics,
    _normalise_macro_builder,
    _projection_pca,
)
from prism.utils.io import save_csv, save_json
from prism.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepSpec:
    couplings: tuple[float, ...]
    seeds: tuple[int, ...]
    obs_designs: tuple[str, ...]
    builders: tuple[str, ...]
    eps_macros: tuple[float, ...]
    length: int
    train_frac: float
    macro_dim: int
    macro_bins: int
    em_iters: int
    obs_dim: int
    pca_kmeans_ks: tuple[int, ...]
    slow_bins: int = 3
    phase_bins: int = 6


def _kmeans(
    values: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
    max_iter: int = 100,
) -> np.ndarray:
    """Minimal Lloyd loop for the PCA baseline."""
    n_samples = values.shape[0]
    if n_clusters <= 0:
        raise ValueError("n_clusters must be >= 1.")
    if n_samples == 0:
        return np.zeros((0,), dtype=int)
    if n_clusters == 1 or n_samples <= n_clusters:
        return np.zeros(n_samples, dtype=int) if n_clusters == 1 else np.arange(n_samples, dtype=int)

    rng = np.random.default_rng(seed)
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


def _macro_obs_for_pca(y_train: np.ndarray, macro_dim: int) -> np.ndarray:
    projection = _projection_pca(y_train, macro_dim)
    return y_train @ projection.T


def _block_attribution_for_window(
    latent: np.ndarray,
    t_start: int,
    t_stop: int,
) -> np.ndarray:
    sub = latent[t_start:t_stop]
    half = sub.shape[1] // 2
    first_norm = (sub[:, :half] ** 2).sum(axis=1)
    second_norm = (sub[:, half:] ** 2).sum(axis=1)
    return (second_norm > first_norm).astype(int)


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


def _append_progress(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
        handle.flush()


def run_sweep(spec: SweepSpec, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "recovery.csv"
    if csv_path.exists():
        csv_path.unlink()
    progress_path = outdir / "progress.log"
    progress_path.write_text("", encoding="utf-8")

    fieldnames = [
        "coupling",
        "seed",
        "obs_design",
        "method",
        "method_param",
        "n_states",
        "ari_block_attribution",
        "ari_slow_block",
        "ari_phase_block",
        "ari_joint",
        "ari_ground_truth",
        "macro_logloss_test",
        "gaussian_logloss_test",
        "unifilarity",
        "branch_entropy",
        "macro_build_time_s",
    ]

    rows: list[dict[str, object]] = []
    total_runs = (
        len(spec.couplings)
        * len(spec.seeds)
        * len(spec.obs_designs)
        * (len(spec.builders) * len(spec.eps_macros) + len(spec.pca_kmeans_ks) + 1)
    )
    completed = 0
    sweep_start = time.perf_counter()
    LOGGER.info(
        "Sweep start | %d runs across %d generator settings",
        total_runs,
        len(spec.couplings) * len(spec.seeds) * len(spec.obs_designs),
    )
    _append_progress(
        progress_path,
        (
            f"sweep start | total_runs={total_runs} "
            f"generator_settings={len(spec.couplings) * len(spec.seeds) * len(spec.obs_designs)}"
        ),
    )

    for coupling in spec.couplings:
        for obs_design in spec.obs_designs:
            for seed in spec.seeds:
                setting_start = time.perf_counter()
                _append_progress(
                    progress_path,
                    f"setting start | coupling={coupling:g} seed={seed} obs_design={obs_design}",
                )
                generator = BlockModularLGSSM(
                    coupling=coupling,
                    obs_dim=spec.obs_dim,
                    obs_design=obs_design,
                )
                sample = generator.sample(length=spec.length, seed=seed)
                obs = np.asarray(sample.x, dtype=float)
                if obs.ndim == 1:
                    obs = obs.reshape(-1, 1)
                latent = np.asarray(sample.latent, dtype=float)
                ground_truth_full = generator.block_attribution(latent)

                split = max(1, min(int(spec.length * spec.train_frac), spec.length - 1))
                y_train = obs[:split]
                y_test = obs[split:]

                iss_cfg = KalmanISSConfig(
                    latent_dim=int(np.sum(generator.block_dims)),
                    em_iters=spec.em_iters,
                    seed=seed,
                )
                _append_progress(
                    progress_path,
                    f"em start | coupling={coupling:g} seed={seed} obs_design={obs_design}",
                )
                iss_model = fit_kalman_iss_em(y_train, iss_cfg)
                _append_progress(
                    progress_path,
                    f"em done | coupling={coupling:g} seed={seed} obs_design={obs_design}",
                )
                steady_solution = solve_steady_state_kalman(iss_model, strict=False)
                mu_pred_test, cov_pred_test, _ = one_step_predictive_y(
                    obs,
                    iss_model,
                    steady_state=True,
                    steady_state_solution=steady_solution,
                )
                mu_pred_test = mu_pred_test[split:]
                cov_pred_test = cov_pred_test[split:]
                gauss_nll = _gaussian_held_out_nll(y_test, mu_pred_test, cov_pred_test)

                projection = _projection_pca(y_train, spec.macro_dim)

                regime_full = generator.regime_labels(
                    latent,
                    slow_bins=spec.slow_bins,
                    phase_bins=spec.phase_bins,
                )
                gt_train = _block_attribution_for_window(latent, 1, split)
                regime_train = {name: values[1:split] for name, values in regime_full.items()}
                ground_truths_train = {
                    "block_attribution": gt_train,
                    "slow_block": regime_train["slow_block"],
                    "phase_block": regime_train["phase_block"],
                    "joint": regime_train["joint"],
                }

                label_path = outdir / f"labels_eps{coupling:g}_seed{seed}_{obs_design}.npz"
                npz_payload: dict[str, np.ndarray] = {
                    "ground_truth_full": ground_truth_full,
                    "ground_truth_train": gt_train,
                    "gt_slow_block_full": regime_full["slow_block"],
                    "gt_phase_block_full": regime_full["phase_block"],
                    "gt_joint_full": regime_full["joint"],
                    "split": np.asarray([split], dtype=int),
                }

                for builder in spec.builders:
                    for eps_macro in spec.eps_macros:
                        start = time.perf_counter()
                        _append_progress(
                            progress_path,
                            (
                                f"macro start | completed={completed}/{total_runs} "
                                f"coupling={coupling:g} seed={seed} obs_design={obs_design} "
                                f"builder={builder} eps_macro={eps_macro:g}"
                            ),
                        )
                        try:
                            macro = _build_macro_dynamics(
                                y_train=y_train,
                                iss_model=iss_model,
                                projection=projection,
                                eps=float(eps_macro),
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
                        except Exception as exc:
                            LOGGER.warning(
                                "builder=%s eps=%g failed for coupling=%g seed=%d %s: %s",
                                builder,
                                eps_macro,
                                coupling,
                                seed,
                                obs_design,
                                exc,
                            )
                            completed += 1
                            _append_progress(
                                progress_path,
                                (
                                    f"macro failed | completed={completed}/{total_runs} "
                                    f"coupling={coupling:g} seed={seed} obs_design={obs_design} "
                                    f"builder={builder} eps_macro={eps_macro:g} error={exc}"
                                ),
                            )
                            continue

                        labels = np.asarray(macro.labels, dtype=int)
                        n_pred = labels.shape[0]
                        aris = {
                            key: float(_adjusted_rand_index(labels, truth[:n_pred]))
                            for key, truth in ground_truths_train.items()
                        }
                        rows.append(
                            {
                                "coupling": float(coupling),
                                "seed": int(seed),
                                "obs_design": obs_design,
                                "method": f"iss_{_normalise_macro_builder(builder)}",
                                "method_param": float(eps_macro),
                                "n_states": int(macro.n_macro_states),
                                "ari_block_attribution": aris["block_attribution"],
                                "ari_slow_block": aris["slow_block"],
                                "ari_phase_block": aris["phase_block"],
                                "ari_joint": aris["joint"],
                                "ari_ground_truth": max(aris.values()),
                                "macro_logloss_test": math.nan,
                                "gaussian_logloss_test": gauss_nll,
                                "unifilarity": (
                                    float(macro.unifilarity)
                                    if math.isfinite(macro.unifilarity)
                                    else math.nan
                                ),
                                "branch_entropy": (
                                    float(macro.branch_entropy)
                                    if math.isfinite(macro.branch_entropy)
                                    else math.nan
                                ),
                                "macro_build_time_s": float(time.perf_counter() - start),
                            }
                        )
                        npz_payload[
                            f"iss_{_normalise_macro_builder(builder)}_eps{eps_macro:g}"
                        ] = labels
                        completed += 1
                        _append_progress(
                            progress_path,
                            (
                                f"macro done | completed={completed}/{total_runs} "
                                f"coupling={coupling:g} seed={seed} obs_design={obs_design} "
                                f"builder={builder} eps_macro={eps_macro:g} "
                                f"states={macro.n_macro_states} elapsed={time.perf_counter() - start:.2f}s"
                            ),
                        )
                        LOGGER.info(
                            "Progress %d/%d | coupling=%g seed=%d %s builder=%s eps=%g states=%d "
                            "ARI(block/slow/phase/joint)=%.2f/%.2f/%.2f/%.2f",
                            completed,
                            total_runs,
                            coupling,
                            seed,
                            obs_design,
                            builder,
                            eps_macro,
                            macro.n_macro_states,
                            aris["block_attribution"],
                            aris["slow_block"],
                            aris["phase_block"],
                            aris["joint"],
                        )

                macro_obs = _macro_obs_for_pca(y_train, spec.macro_dim)
                gt_for_pca = gt_train
                if macro_obs.shape[0] >= gt_for_pca.shape[0] + 1:
                    macro_obs_aligned = macro_obs[1 : 1 + gt_for_pca.shape[0]]
                else:
                    macro_obs_aligned = macro_obs[: gt_for_pca.shape[0]]

                for n_clusters in spec.pca_kmeans_ks:
                    start = time.perf_counter()
                    _append_progress(
                        progress_path,
                        (
                            f"pca_kmeans start | completed={completed}/{total_runs} "
                            f"coupling={coupling:g} seed={seed} obs_design={obs_design} k={n_clusters}"
                        ),
                    )
                    km_labels = _kmeans(macro_obs_aligned, int(n_clusters), seed=seed)
                    n_pred = km_labels.shape[0]
                    aris = {
                        key: float(_adjusted_rand_index(km_labels, truth[:n_pred]))
                        for key, truth in ground_truths_train.items()
                    }
                    rows.append(
                        {
                            "coupling": float(coupling),
                            "seed": int(seed),
                            "obs_design": obs_design,
                            "method": "pca_kmeans",
                            "method_param": float(n_clusters),
                            "n_states": int(n_clusters),
                            "ari_block_attribution": aris["block_attribution"],
                            "ari_slow_block": aris["slow_block"],
                            "ari_phase_block": aris["phase_block"],
                            "ari_joint": aris["joint"],
                            "ari_ground_truth": max(aris.values()),
                            "macro_logloss_test": math.nan,
                            "gaussian_logloss_test": gauss_nll,
                            "unifilarity": math.nan,
                            "branch_entropy": math.nan,
                            "macro_build_time_s": float(time.perf_counter() - start),
                        }
                    )
                    npz_payload[f"pca_kmeans_k{int(n_clusters)}"] = km_labels
                    completed += 1
                    _append_progress(
                        progress_path,
                        (
                            f"pca_kmeans done | completed={completed}/{total_runs} "
                            f"coupling={coupling:g} seed={seed} obs_design={obs_design} "
                            f"k={n_clusters} elapsed={time.perf_counter() - start:.2f}s"
                        ),
                    )
                    LOGGER.info(
                        "Progress %d/%d | coupling=%g seed=%d %s pca_kmeans k=%d "
                        "ARI(block/slow/phase/joint)=%.2f/%.2f/%.2f/%.2f",
                        completed,
                        total_runs,
                        coupling,
                        seed,
                        obs_design,
                        n_clusters,
                        aris["block_attribution"],
                        aris["slow_block"],
                        aris["phase_block"],
                        aris["joint"],
                    )

                rows.append(
                    {
                        "coupling": float(coupling),
                        "seed": int(seed),
                        "obs_design": obs_design,
                        "method": "ground_truth_majority",
                        "method_param": math.nan,
                        "n_states": 2,
                        "ari_block_attribution": 0.0,
                        "ari_slow_block": 0.0,
                        "ari_phase_block": 0.0,
                        "ari_joint": 0.0,
                        "ari_ground_truth": 0.0,
                        "macro_logloss_test": math.nan,
                        "gaussian_logloss_test": gauss_nll,
                        "unifilarity": math.nan,
                        "branch_entropy": math.nan,
                        "macro_build_time_s": 0.0,
                    }
                )
                completed += 1
                np.savez(label_path, **npz_payload)
                _append_progress(
                    progress_path,
                    (
                        f"setting done | completed={completed}/{total_runs} "
                        f"coupling={coupling:g} seed={seed} obs_design={obs_design} "
                        f"label_file={label_path.name} elapsed={time.perf_counter() - setting_start:.2f}s"
                    ),
                )

    save_csv(csv_path, rows, append=False, fieldnames=fieldnames)
    save_json(
        outdir / "sweep_spec.json",
        {
            "couplings": list(spec.couplings),
            "seeds": list(spec.seeds),
            "obs_designs": list(spec.obs_designs),
            "builders": list(spec.builders),
            "eps_macros": list(spec.eps_macros),
            "length": spec.length,
            "train_frac": spec.train_frac,
            "macro_dim": spec.macro_dim,
            "macro_bins": spec.macro_bins,
            "em_iters": spec.em_iters,
            "obs_dim": spec.obs_dim,
            "pca_kmeans_ks": list(spec.pca_kmeans_ks),
            "slow_bins": spec.slow_bins,
            "phase_bins": spec.phase_bins,
        },
    )
    LOGGER.info(
        "Sweep complete | rows=%d csv=%s elapsed=%.1fs",
        len(rows),
        csv_path,
        time.perf_counter() - sweep_start,
    )
    _append_progress(
        progress_path,
        f"sweep complete | rows={len(rows)} csv={csv_path.name} elapsed={time.perf_counter() - sweep_start:.1f}s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block-modular LGSSM recovery sweep.")
    parser.add_argument("--couplings", nargs="+", type=float, default=[0.0, 0.025, 0.05, 0.10, 0.20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--obs-designs",
        nargs="+",
        type=str,
        choices=["random", "aligned"],
        default=["random", "aligned"],
    )
    parser.add_argument(
        "--builders",
        nargs="+",
        type=str,
        default=["hierarchical_single", "hierarchical_complete", "linear_quantile", "greedy"],
    )
    parser.add_argument("--eps-macros", nargs="+", type=float, default=[0.10, 0.15, 0.25, 0.40])
    parser.add_argument("--length", type=int, default=4000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--macro-dim", type=int, default=2)
    parser.add_argument("--macro-bins", type=int, default=3)
    parser.add_argument("--em-iters", type=int, default=50)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--pca-kmeans-ks", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--slow-bins", type=int, default=3)
    parser.add_argument("--phase-bins", type=int, default=6)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging(getattr(logging, args.log_level))
    spec = SweepSpec(
        couplings=tuple(args.couplings),
        seeds=tuple(args.seeds),
        obs_designs=tuple(args.obs_designs),
        builders=tuple(args.builders),
        eps_macros=tuple(args.eps_macros),
        length=int(args.length),
        train_frac=float(args.train_frac),
        macro_dim=int(args.macro_dim),
        macro_bins=int(args.macro_bins),
        em_iters=int(args.em_iters),
        obs_dim=int(args.obs_dim),
        pca_kmeans_ks=tuple(args.pca_kmeans_ks),
        slow_bins=int(args.slow_bins),
        phase_bins=int(args.phase_bins),
    )
    run_sweep(spec, args.outdir)


if __name__ == "__main__":
    main()
