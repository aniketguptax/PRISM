from __future__ import annotations

import argparse
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prism.processes.hierarchical_predictive_hmm import HierarchicalPredictiveHMM
from prism.reconstruction.kalman_iss import _adjusted_rand_index
from prism.utils.io import save_csv, save_json


@dataclass(frozen=True)
class SweepSpec:
    noises: tuple[float, ...]
    seeds: tuple[int, ...]
    eps_values: tuple[float, ...]
    kmeans_ks: tuple[int, ...]
    history_clusterers: tuple[str, ...]
    length: int
    train_frac: float
    context_len: int
    future_horizon: int
    min_context_count: int
    make_plots: bool


def _append_progress(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
        handle.flush()


def _context_key(x: np.ndarray, t: int, context_len: int) -> tuple[int, ...]:
    return tuple(int(v) for v in x[t - context_len + 1 : t + 1].tolist())


def _context_vector(key: tuple[int, ...], alphabet_size: int) -> np.ndarray:
    out = np.zeros((len(key), alphabet_size), dtype=float)
    for idx, symbol in enumerate(key):
        out[idx, int(symbol)] = 1.0
    return out.reshape(-1)


def _predictive_signatures(
    x: np.ndarray,
    *,
    context_len: int,
    future_horizon: int,
    alphabet_size: int,
    min_count: int,
) -> tuple[dict[tuple[int, ...], np.ndarray], np.ndarray]:
    counts: dict[tuple[int, ...], np.ndarray] = defaultdict(
        lambda: np.zeros((future_horizon, alphabet_size), dtype=float)
    )
    support: Counter[tuple[int, ...]] = Counter()
    for t in range(context_len - 1, x.shape[0] - future_horizon):
        key = _context_key(x, t, context_len)
        support[key] += 1
        for lag in range(1, future_horizon + 1):
            counts[key][lag - 1, int(x[t + lag])] += 1.0

    global_counts = np.zeros((future_horizon, alphabet_size), dtype=float)
    for value in counts.values():
        global_counts += value
    global_signature = _normalise_rows(global_counts)

    signatures: dict[tuple[int, ...], np.ndarray] = {}
    for key, value in counts.items():
        if support[key] < min_count:
            continue
        signatures[key] = _normalise_rows(value).reshape(-1)
    return signatures, global_signature.reshape(-1)


def _normalise_rows(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    fallback = np.full(counts.shape[1], 1.0 / counts.shape[1])
    probs = np.broadcast_to(fallback, counts.shape).copy()
    np.divide(counts, totals, out=probs, where=totals > 0.0)
    return probs


def _single_link_labels(signatures: np.ndarray, eps: float) -> np.ndarray:
    n_items = signatures.shape[0]
    labels = np.full(n_items, -1, dtype=int)
    cluster = 0
    for start in range(n_items):
        if labels[start] >= 0:
            continue
        labels[start] = cluster
        stack = [start]
        while stack:
            idx = stack.pop()
            distances = np.linalg.norm(signatures - signatures[idx], axis=1)
            neighbours = np.where((distances <= eps) & (labels < 0))[0]
            for neighbour in neighbours.tolist():
                labels[neighbour] = cluster
                stack.append(int(neighbour))
        cluster += 1
    return labels


def _kmeans(
    values: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = values.shape[0]
    if n_clusters <= 0:
        raise ValueError("n_clusters must be >= 1.")
    if n_samples == 0:
        return np.zeros((0,), dtype=int), np.zeros((n_clusters, values.shape[1]), dtype=float)
    if n_clusters == 1:
        return np.zeros(n_samples, dtype=int), values[:1].copy()

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
    return labels, centres


def _agglomerative_labels(values: np.ndarray, n_clusters: int, *, linkage: str) -> np.ndarray:
    n_samples = values.shape[0]
    if n_clusters <= 0:
        raise ValueError("n_clusters must be >= 1.")
    if n_samples == 0:
        return np.zeros((0,), dtype=int)
    n_clusters = min(n_clusters, n_samples)
    clusters: list[list[int]] = [[idx] for idx in range(n_samples)]

    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)

    def cluster_distance(a: list[int], b: list[int]) -> float:
        block = distances[np.ix_(a, b)]
        if linkage == "single":
            return float(np.min(block))
        if linkage == "complete":
            return float(np.max(block))
        if linkage == "average":
            return float(np.mean(block))
        raise ValueError(f"Unknown linkage={linkage!r}")

    while len(clusters) > n_clusters:
        best_pair: tuple[int, int] | None = None
        best_distance = math.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = cluster_distance(clusters[i], clusters[j])
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]

    labels = np.zeros(n_samples, dtype=int)
    for label, cluster in enumerate(clusters):
        labels[cluster] = label
    return labels


def _transition_diagnostics(labels: np.ndarray, symbols: np.ndarray) -> tuple[float, float]:
    counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for idx in range(labels.shape[0] - 1):
        counts[(int(labels[idx]), int(symbols[idx + 1]))][int(labels[idx + 1])] += 1
    total = sum(sum(counter.values()) for counter in counts.values())
    if total <= 0:
        return math.nan, math.nan

    unif = 0.0
    branch = 0.0
    for counter in counts.values():
        denom = sum(counter.values())
        if denom <= 0:
            continue
        probs = np.asarray([value / denom for value in counter.values()], dtype=float)
        weight = denom / total
        unif += weight * float(probs.max())
        branch += weight * float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
    return unif, branch


def _cluster_next_symbol_probs(
    labels: np.ndarray,
    x: np.ndarray,
    times: np.ndarray,
    *,
    alphabet_size: int,
) -> np.ndarray:
    n_states = int(labels.max()) + 1 if labels.size else 1
    counts = np.full((n_states, alphabet_size), 1e-3, dtype=float)
    next_times = times + 1
    valid = next_times < x.shape[0]
    if np.any(valid):
        np.add.at(
            counts,
            (labels[valid].astype(int), x[next_times[valid]].astype(int)),
            1.0,
        )
    return counts / counts.sum(axis=1, keepdims=True)


def _ensure_probability_rows(
    probs: np.ndarray,
    *,
    min_rows: int,
    alphabet_size: int,
) -> np.ndarray:
    if probs.shape[0] >= min_rows:
        return probs

    padded = np.empty((min_rows, alphabet_size), dtype=float)
    padded[: probs.shape[0]] = probs
    padded[probs.shape[0] :] = 1.0 / alphabet_size
    return padded


def _logloss(
    labels: np.ndarray,
    x: np.ndarray,
    times: np.ndarray,
    probs: np.ndarray,
) -> float:
    if labels.size == 0:
        return math.nan
    total = 0.0
    n = 0
    for label, t in zip(labels.tolist(), times.tolist()):
        if t + 1 >= x.shape[0]:
            continue
        total -= math.log(max(float(probs[int(label), int(x[t + 1])]), 1e-12))
        n += 1
    return float(total / n) if n else math.nan


def _labels_for_predictive_clusters(
    x: np.ndarray,
    times: np.ndarray,
    *,
    context_len: int,
    context_to_cluster: dict[tuple[int, ...], int],
    fallback_label: int,
) -> np.ndarray:
    labels = np.zeros(times.shape[0], dtype=int)
    for idx, t in enumerate(times.tolist()):
        key = _context_key(x, int(t), context_len)
        labels[idx] = context_to_cluster.get(key, fallback_label)
    return labels


def _labels_for_kmeans(
    x: np.ndarray,
    times: np.ndarray,
    *,
    context_len: int,
    alphabet_size: int,
    centres: np.ndarray,
) -> np.ndarray:
    times = np.asarray(times, dtype=int)
    offsets = np.arange(context_len - 1, -1, -1, dtype=int)
    symbols = x[times[:, None] - offsets[None, :]].astype(int)
    values = np.zeros((times.shape[0], context_len, alphabet_size), dtype=float)
    values[
        np.arange(times.shape[0])[:, None],
        np.arange(context_len, dtype=int)[None, :],
        symbols,
    ] = 1.0
    values = values.reshape(times.shape[0], -1)
    distances = np.linalg.norm(values[:, None, :] - centres[None, :, :], axis=-1)
    return np.argmin(distances, axis=1).astype(int)


def _plot_summary(rows: list[dict[str, object]], outdir: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    methods = sorted({str(row["method"]) for row in rows})
    metrics = [
        ("n_states", "State count |M|"),
        ("ari_coarse", "ARI vs coarse regime"),
        ("ari_fine", "ARI vs fine state"),
        ("ari_joint", "ARI vs joint state"),
        ("test_logloss", "Held-out next-symbol NLL"),
        ("unifilarity", "Unifilarity"),
        ("branch_entropy", "Branch entropy (bits)"),
    ]
    noises = sorted({float(row["noise"]) for row in rows})
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        len(metrics),
        len(methods),
        figsize=(3.1 * len(methods), 2.2 * len(metrics)),
        squeeze=False,
    )
    for col, method in enumerate(methods):
        method_rows = [row for row in rows if row["method"] == method]
        params = sorted({float(row["method_param"]) for row in method_rows})
        for row_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx][col]
            for noise_idx, noise in enumerate(noises):
                ys: list[float] = []
                for param in params:
                    values = [
                        float(row[metric])
                        for row in method_rows
                        if abs(float(row["noise"]) - noise) < 1e-12
                        and abs(float(row["method_param"]) - param) < 1e-12
                        and math.isfinite(float(row[metric]))
                    ]
                    ys.append(float(np.mean(values)) if values else math.nan)
                ax.plot(
                    params,
                    ys,
                    marker="o",
                    color=cmap(noise_idx / max(len(noises) - 1, 1)),
                    label=f"noise={noise:g}",
                )
            if row_idx == 0:
                ax.set_title(method, fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("eps or k", fontsize=8)
            ax.tick_params(labelsize=7)
            if metric == "n_states":
                ax.set_yscale("log")

    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(noises), fontsize=7)
    fig.suptitle("Hierarchical predictive-state recovery", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(figures / "hierarchical_predictive_recovery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_sweep(spec: SweepSpec, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    progress_path = outdir / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    rows: list[dict[str, object]] = []

    n_history_runs = len(spec.kmeans_ks) * len(spec.history_clusterers)
    total = len(spec.noises) * len(spec.seeds) * (len(spec.eps_values) + n_history_runs)
    completed = 0
    started = time.perf_counter()
    _append_progress(progress_path, f"sweep start | total_runs={total}")

    for noise in spec.noises:
        for seed in spec.seeds:
            setting_start = time.perf_counter()
            _append_progress(progress_path, f"setting start | noise={noise:g} seed={seed}")
            generator = HierarchicalPredictiveHMM(emission_noise=noise)
            sample = generator.sample(spec.length, seed=seed)
            x = np.asarray(sample.x, dtype=int)
            latent = np.asarray(sample.latent, dtype=int)
            regimes = generator.regime_labels(latent)

            split = max(spec.context_len + spec.future_horizon + 1, int(spec.length * spec.train_frac))
            split = min(split, spec.length - spec.future_horizon - 2)
            train_x = x[:split]
            alphabet_size = generator.alphabet_size
            train_times = np.arange(spec.context_len - 1, split - spec.future_horizon, dtype=int)
            test_times = np.arange(split, spec.length - spec.future_horizon, dtype=int)

            signatures, global_signature = _predictive_signatures(
                train_x,
                context_len=spec.context_len,
                future_horizon=spec.future_horizon,
                alphabet_size=alphabet_size,
                min_count=spec.min_context_count,
            )
            context_keys = sorted(signatures)
            signature_matrix = (
                np.asarray([signatures[key] for key in context_keys], dtype=float)
                if context_keys
                else global_signature.reshape(1, -1)
            )

            npz_payload: dict[str, np.ndarray] = {
                "x": x,
                "coarse": regimes["coarse"],
                "fine": regimes["fine"],
                "joint": regimes["joint"],
                "split": np.asarray([split], dtype=int),
            }

            for eps in spec.eps_values:
                start = time.perf_counter()
                cluster_labels = _single_link_labels(signature_matrix, float(eps))
                context_to_cluster = {
                    key: int(label)
                    for key, label in zip(context_keys, cluster_labels.tolist())
                }
                fallback_label = int(cluster_labels.max()) + 1 if cluster_labels.size else 0
                labels_train = _labels_for_predictive_clusters(
                    x,
                    train_times,
                    context_len=spec.context_len,
                    context_to_cluster=context_to_cluster,
                    fallback_label=fallback_label,
                )
                labels_test = _labels_for_predictive_clusters(
                    x,
                    test_times,
                    context_len=spec.context_len,
                    context_to_cluster=context_to_cluster,
                    fallback_label=fallback_label,
                )
                probs = _cluster_next_symbol_probs(
                    labels_train,
                    x,
                    train_times,
                    alphabet_size=alphabet_size,
                )
                if fallback_label >= probs.shape[0]:
                    probs = _ensure_probability_rows(
                        probs,
                        min_rows=fallback_label + 1,
                        alphabet_size=alphabet_size,
                    )
                unif, branch = _transition_diagnostics(labels_train, x[train_times])
                n_states = int(max(labels_train.max(), labels_test.max()) + 1)
                row = {
                    "noise": float(noise),
                    "seed": int(seed),
                    "method": "prism_predictive",
                    "method_param": float(eps),
                    "n_states": n_states,
                    "ari_coarse": float(_adjusted_rand_index(labels_train, regimes["coarse"][train_times])),
                    "ari_fine": float(_adjusted_rand_index(labels_train, regimes["fine"][train_times])),
                    "ari_joint": float(_adjusted_rand_index(labels_train, regimes["joint"][train_times])),
                    "test_logloss": _logloss(labels_test, x, test_times, probs),
                    "unifilarity": unif,
                    "branch_entropy": branch,
                    "elapsed_s": float(time.perf_counter() - start),
                }
                rows.append(row)
                npz_payload[f"prism_predictive_eps{eps:g}"] = labels_train
                completed += 1
                _append_progress(
                    progress_path,
                    (
                        f"method done | completed={completed}/{total} noise={noise:g} "
                        f"seed={seed} method=prism_predictive eps={eps:g} "
                        f"states={n_states} ari_coarse={row['ari_coarse']:.3f}"
                    ),
                )

            context_values = np.asarray(
                [
                    _context_vector(_context_key(x, int(t), spec.context_len), alphabet_size)
                    for t in train_times.tolist()
                ],
                dtype=float,
            )
            train_context_keys = [
                _context_key(x, int(t), spec.context_len) for t in train_times.tolist()
            ]
            unique_context_keys = sorted(set(train_context_keys))
            unique_context_values = np.asarray(
                [_context_vector(key, alphabet_size) for key in unique_context_keys],
                dtype=float,
            )
            for k in spec.kmeans_ks:
                for clusterer in spec.history_clusterers:
                    start = time.perf_counter()
                    method = f"history_{clusterer}"
                    if clusterer == "kmeans":
                        labels_train, centres = _kmeans(context_values, int(k), seed=seed)
                        labels_test = _labels_for_kmeans(
                            x,
                            test_times,
                            context_len=spec.context_len,
                            alphabet_size=alphabet_size,
                            centres=centres,
                        )
                    else:
                        unique_labels = _agglomerative_labels(
                            unique_context_values,
                            int(k),
                            linkage=clusterer,
                        )
                        context_to_cluster = {
                            key: int(label)
                            for key, label in zip(unique_context_keys, unique_labels.tolist())
                        }
                        fallback_label = int(unique_labels.max()) + 1 if unique_labels.size else 0
                        labels_train = _labels_for_predictive_clusters(
                            x,
                            train_times,
                            context_len=spec.context_len,
                            context_to_cluster=context_to_cluster,
                            fallback_label=fallback_label,
                        )
                        labels_test = _labels_for_predictive_clusters(
                            x,
                            test_times,
                            context_len=spec.context_len,
                            context_to_cluster=context_to_cluster,
                            fallback_label=fallback_label,
                        )
                    probs = _cluster_next_symbol_probs(
                        labels_train,
                        x,
                        train_times,
                        alphabet_size=alphabet_size,
                    )
                    if labels_test.size and labels_test.max() >= probs.shape[0]:
                        probs = _ensure_probability_rows(
                            probs,
                            min_rows=int(labels_test.max()) + 1,
                            alphabet_size=alphabet_size,
                        )
                    unif, branch = _transition_diagnostics(labels_train, x[train_times])
                    row = {
                        "noise": float(noise),
                        "seed": int(seed),
                        "method": method,
                        "method_param": float(k),
                        "n_states": int(max(labels_train.max(), labels_test.max()) + 1),
                        "ari_coarse": float(_adjusted_rand_index(labels_train, regimes["coarse"][train_times])),
                        "ari_fine": float(_adjusted_rand_index(labels_train, regimes["fine"][train_times])),
                        "ari_joint": float(_adjusted_rand_index(labels_train, regimes["joint"][train_times])),
                        "test_logloss": _logloss(labels_test, x, test_times, probs),
                        "unifilarity": unif,
                        "branch_entropy": branch,
                        "elapsed_s": float(time.perf_counter() - start),
                    }
                    rows.append(row)
                    npz_payload[f"{method}_k{int(k)}"] = labels_train
                    completed += 1
                    _append_progress(
                        progress_path,
                        (
                            f"method done | completed={completed}/{total} noise={noise:g} "
                            f"seed={seed} method={method} k={k} "
                            f"ari_coarse={row['ari_coarse']:.3f}"
                        ),
                    )

            np.savez(outdir / f"labels_noise{noise:g}_seed{seed}.npz", **npz_payload)
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
        "test_logloss",
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
            "eps_values": list(spec.eps_values),
            "kmeans_ks": list(spec.kmeans_ks),
            "history_clusterers": list(spec.history_clusterers),
            "length": spec.length,
            "train_frac": spec.train_frac,
            "context_len": spec.context_len,
            "future_horizon": spec.future_horizon,
            "min_context_count": spec.min_context_count,
            "make_plots": spec.make_plots,
        },
    )
    if spec.make_plots:
        _plot_summary(rows, outdir)
    _append_progress(
        progress_path,
        f"sweep complete | rows={len(rows)} csv=recovery.csv elapsed={time.perf_counter() - started:.1f}s",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noises", nargs="+", type=float, default=[0.02, 0.08, 0.16, 0.28])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eps-values", nargs="+", type=float, default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--kmeans-ks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument(
        "--history-clusterers",
        nargs="+",
        choices=["kmeans", "single", "complete", "average"],
        default=["kmeans", "complete", "average"],
    )
    parser.add_argument("--length", type=int, default=12000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--context-len", type=int, default=2)
    parser.add_argument("--future-horizon", type=int, default=4)
    parser.add_argument("--min-context-count", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib figure generation.")
    parser.add_argument("--outdir", type=Path, default=Path("./results/hierarchical_predictive_sweep"))
    args = parser.parse_args()

    if args.length < args.context_len + args.future_horizon + 10:
        raise ValueError("--length is too small for the chosen context/future horizons.")
    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("--train-frac must lie in (0, 1).")

    spec = SweepSpec(
        noises=tuple(args.noises),
        seeds=tuple(args.seeds),
        eps_values=tuple(args.eps_values),
        kmeans_ks=tuple(args.kmeans_ks),
        history_clusterers=tuple(args.history_clusterers),
        length=int(args.length),
        train_frac=float(args.train_frac),
        context_len=int(args.context_len),
        future_horizon=int(args.future_horizon),
        min_context_count=int(args.min_context_count),
        make_plots=not bool(args.no_plots),
    )
    run_sweep(spec, args.outdir)


if __name__ == "__main__":
    main()
