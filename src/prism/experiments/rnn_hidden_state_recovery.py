from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prism.experiments.hierarchical_predictive_recovery import (
    _cluster_next_symbol_probs,
    _kmeans,
    _logloss,
    _transition_diagnostics,
)
from prism.processes.even_process import EvenProcess
from prism.processes.hierarchical_predictive_hmm import HierarchicalPredictiveHMM
from prism.reconstruction.kalman_iss import _adjusted_rand_index
from prism.utils.io import save_csv, save_json


@dataclass(frozen=True)
class ExperimentConfig:
    process: str
    train_length: int
    test_length: int
    emission_noise: float
    even_p: float
    modulo_n: int
    modulo_reset_prob: float
    seed: int
    hidden_dim: int
    embedding_dim: int
    unroll: int
    batch_size: int
    epochs: int
    batches_per_epoch: int
    learning_rate: float
    burn_in: int
    rollout_horizon: int
    cluster_ks: tuple[int, ...]
    kmeans_n_init: int
    context_len: int
    ablation_horizons: tuple[int, ...]
    ablation_context_lens: tuple[int, ...]
    make_plots: bool


def _torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "rnn_hidden_state_recovery requires torch. Use the project venv or install requirements.txt."
        ) from exc
    return torch


class GRUPredictor:
    def __init__(self, alphabet_size: int, embedding_dim: int, hidden_dim: int, *, seed: int) -> None:
        torch = _torch()
        torch.manual_seed(seed)
        self.torch = torch
        self.model = _GRUNextSymbolModel(alphabet_size, embedding_dim, hidden_dim)

    def train(
        self,
        x: np.ndarray,
        *,
        epochs: int,
        batches_per_epoch: int,
        batch_size: int,
        unroll: int,
        learning_rate: float,
        seed: int,
    ) -> list[dict[str, float]]:
        torch = self.torch
        if x.shape[0] <= unroll + 1:
            raise ValueError("Training sequence is shorter than the requested unroll length.")

        rng = np.random.default_rng(seed)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        history: list[dict[str, float]] = []
        max_start = x.shape[0] - unroll - 1

        self.model.train()
        started = time.perf_counter()
        for epoch in range(epochs):
            losses: list[float] = []
            for _ in range(batches_per_epoch):
                starts = rng.integers(0, max_start + 1, size=batch_size)
                inputs = np.stack([x[start : start + unroll] for start in starts], axis=0)
                targets = np.stack([x[start + 1 : start + unroll + 1] for start in starts], axis=0)

                input_t = torch.as_tensor(inputs, dtype=torch.long)
                target_t = torch.as_tensor(targets, dtype=torch.long)
                logits, _ = self.model(input_t)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_t.reshape(-1))

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimiser.step()
                losses.append(float(loss.detach().cpu().item()))

            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_nll": float(np.mean(losses)),
                    "elapsed_s": float(time.perf_counter() - started),
                }
            )
        return history

    def extract(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            input_t = torch.as_tensor(x[:-1][None, :], dtype=torch.long)
            logits, hidden_seq = self.model(input_t)
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            hidden = hidden_seq[0].cpu().numpy()
        return hidden.astype(float), probs.astype(float)

    def rollout_signatures(
        self,
        hidden: np.ndarray,
        *,
        horizon: int,
        batch_size: int = 512,
    ) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, hidden.shape[0], batch_size):
                block = torch.as_tensor(hidden[start : start + batch_size], dtype=torch.float32)
                sig = self.model.rollout_signature(block, horizon=horizon)
                outputs.append(sig.cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(float)


class _GRUNextSymbolModel:
    def __init__(self, alphabet_size: int, embedding_dim: int, hidden_dim: int) -> None:
        torch = _torch()
        nn = torch.nn
        self.alphabet_size = int(alphabet_size)
        self.hidden_dim = int(hidden_dim)
        self.embedding = nn.Embedding(alphabet_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, alphabet_size)

    def parameters(self):
        return list(self.embedding.parameters()) + list(self.gru.parameters()) + list(self.readout.parameters())

    def train(self) -> None:
        self.embedding.train()
        self.gru.train()
        self.readout.train()

    def eval(self) -> None:
        self.embedding.eval()
        self.gru.eval()
        self.readout.eval()

    def __call__(self, x):
        embedded = self.embedding(x)
        out, _ = self.gru(embedded)
        logits = self.readout(out)
        return logits, out

    def _step(self, symbol, hidden):
        embedded = self.embedding(symbol).unsqueeze(1)
        _, next_hidden = self.gru(embedded, hidden.unsqueeze(0))
        return next_hidden.squeeze(0)

    def rollout_signature(self, hidden, *, horizon: int):
        torch = _torch()
        n = hidden.shape[0]
        signatures = torch.zeros(
            (n, horizon, self.alphabet_size),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        branches: list[tuple[object, object]] = [
            (torch.ones(n, dtype=hidden.dtype, device=hidden.device), hidden)
        ]
        for lag in range(horizon):
            next_branches: list[tuple[object, object]] = []
            for weight, state in branches:
                probs = torch.softmax(self.readout(state), dim=-1)
                signatures[:, lag, :] += weight[:, None] * probs
                if lag == horizon - 1:
                    continue
                for symbol in range(self.alphabet_size):
                    symbol_t = torch.full(
                        (n,),
                        int(symbol),
                        dtype=torch.long,
                        device=hidden.device,
                    )
                    next_state = self._step(symbol_t, state)
                    next_branches.append((weight * probs[:, symbol], next_state))
            branches = next_branches
        return signatures.reshape(n, horizon * self.alphabet_size)


class ModuloRenewalProcess:
    def __init__(self, n_states: int, reset_prob: float) -> None:
        if n_states < 3:
            raise ValueError("ModuloRenewalProcess needs at least three states.")
        if not (0.0 < reset_prob < 1.0):
            raise ValueError("reset_prob must lie in (0, 1).")
        self.n_states = int(n_states)
        self.reset_prob = float(reset_prob)

    def sample(self, length: int, seed: int):
        from prism.processes.protocols import Sample

        rng = np.random.default_rng(seed)
        state = 0
        x: list[int] = []
        latent: list[int] = []
        for _ in range(length):
            latent.append(int(state))
            if state == 0 and rng.random() < self.reset_prob:
                x.append(0)
                state = 0
            else:
                x.append(1)
                state = (state + 1) % self.n_states
        return Sample(x=x, latent=latent)


def _one_hot_history(x: np.ndarray, times: np.ndarray, *, context_len: int, alphabet_size: int) -> np.ndarray:
    offsets = np.arange(context_len - 1, -1, -1, dtype=int)
    symbols = x[times[:, None] - offsets[None, :]].astype(int)
    values = np.zeros((times.shape[0], context_len, alphabet_size), dtype=float)
    values[
        np.arange(times.shape[0])[:, None],
        np.arange(context_len, dtype=int)[None, :],
        symbols,
    ] = 1.0
    return values.reshape(times.shape[0], context_len * alphabet_size)


def _make_process(config: ExperimentConfig):
    if config.process == "hierarchical":
        return HierarchicalPredictiveHMM(emission_noise=config.emission_noise)
    if config.process == "even":
        return EvenProcess(p_emit_one=config.even_p)
    if config.process == "modulo":
        return ModuloRenewalProcess(
            n_states=config.modulo_n,
            reset_prob=config.modulo_reset_prob,
        )
    raise ValueError(f"Unknown process {config.process!r}.")


def _alphabet_size(x: np.ndarray) -> int:
    if x.size == 0:
        raise ValueError("Cannot infer alphabet size from an empty sequence.")
    return int(np.max(x)) + 1


def _label_dict(generator, latent: np.ndarray) -> dict[str, np.ndarray]:
    if hasattr(generator, "regime_labels"):
        return generator.regime_labels(latent)
    labels = np.asarray(latent, dtype=int).reshape(-1)
    if isinstance(generator, (EvenProcess, ModuloRenewalProcess)) and labels.shape[0] > 1:
        labels = np.concatenate([labels[1:], labels[-1:]])
    return {
        "coarse": labels,
        "fine": labels,
        "joint": labels,
    }


def _assign_to_centres(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(values[:, None, :] - centres[None, :, :], axis=-1)
    return np.argmin(distances, axis=1).astype(int)


def _kmeans_best(values: np.ndarray, k: int, *, seed: int, n_init: int) -> tuple[np.ndarray, np.ndarray]:
    best_labels: np.ndarray | None = None
    best_centres: np.ndarray | None = None
    best_inertia = math.inf
    for offset in range(max(1, int(n_init))):
        labels, centres = _kmeans(values, k, seed=seed + 104_729 * offset)
        distances = np.linalg.norm(values - centres[labels], axis=1)
        inertia = float(np.sum(distances * distances))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centres = centres
    if best_labels is None or best_centres is None:
        raise RuntimeError("k-means failed to produce an initialisation.")
    return best_labels, best_centres


def _pca2(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] < 2:
        return np.column_stack([centered[:, 0], np.zeros(centered.shape[0], dtype=float)])
    return centered @ vt[:2].T


def _best_label_map(pred: np.ndarray, true: np.ndarray) -> dict[int, int]:
    pred_vals = sorted(int(v) for v in np.unique(pred))
    true_vals = sorted(int(v) for v in np.unique(true))
    if len(pred_vals) > 8 or len(true_vals) > 8:
        raise ValueError("Brute-force label matching is only intended for small state counts.")

    best_score = -1
    best: dict[int, int] = {}
    for assignment in itertools.permutations(true_vals, min(len(pred_vals), len(true_vals))):
        mapping = {pred_val: true_val for pred_val, true_val in zip(pred_vals, assignment)}
        mapped = np.asarray([mapping.get(int(v), -1) for v in pred], dtype=int)
        score = int(np.sum(mapped == true))
        if score > best_score:
            best_score = score
            best = mapping
    return best


def _modulo_transition_accuracy(
    labels_test: np.ndarray,
    true: np.ndarray,
    x: np.ndarray,
    times: np.ndarray,
    *,
    modulo_n: int,
) -> float:
    mapping = _best_label_map(labels_test, true)
    mapped = np.asarray([mapping.get(int(v), -1) for v in labels_test], dtype=int)
    ok = 0
    total = 0
    for idx in range(mapped.shape[0] - 1):
        current = int(mapped[idx])
        actual_next = int(mapped[idx + 1])
        observed = int(x[int(times[idx]) + 1])
        if current < 0 or actual_next < 0:
            continue
        expected_next = 0 if current == 0 and observed == 0 else (current + 1) % modulo_n
        ok += int(actual_next == expected_next)
        total += 1
    return float(ok / total) if total else math.nan


def _evaluate_cluster_method(
    *,
    method: str,
    k: int,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_times: np.ndarray,
    test_times: np.ndarray,
    train_labels_true: dict[str, np.ndarray],
    test_labels_true: dict[str, np.ndarray],
    seed: int,
    n_init: int,
    alphabet_size: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    labels_train, centres = _kmeans_best(train_features, k, seed=seed, n_init=n_init)
    labels_test = _assign_to_centres(test_features, centres)
    probs = _cluster_next_symbol_probs(
        labels_train,
        train_x,
        train_times,
        alphabet_size=alphabet_size,
    )
    if labels_test.size and labels_test.max() >= probs.shape[0]:
        padded = np.full((int(labels_test.max()) + 1, alphabet_size), 1.0 / alphabet_size, dtype=float)
        padded[: probs.shape[0]] = probs
        probs = padded
    unif, branch = _transition_diagnostics(labels_train, train_x[train_times])
    row: dict[str, object] = {
        "method": method,
        "k": int(k),
        "n_states": int(max(labels_train.max(), labels_test.max()) + 1),
        "ari_coarse": float(_adjusted_rand_index(labels_test, test_labels_true["coarse"])),
        "ari_fine": float(_adjusted_rand_index(labels_test, test_labels_true["fine"])),
        "ari_joint": float(_adjusted_rand_index(labels_test, test_labels_true["joint"])),
        "test_logloss": _logloss(labels_test, test_x, test_times, probs),
        "unifilarity": unif,
        "branch_entropy": branch,
    }
    return row, labels_train, labels_test, centres


def _ablation_row(
    *,
    ablation: str,
    value: int,
    method: str,
    k: int,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_times: np.ndarray,
    test_times: np.ndarray,
    train_labels_true: dict[str, np.ndarray],
    test_labels_true: dict[str, np.ndarray],
    seed: int,
    n_init: int,
    alphabet_size: int,
    modulo_n: int | None,
) -> dict[str, object]:
    row, _, labels_test, _ = _evaluate_cluster_method(
        method=method,
        k=k,
        train_features=train_features,
        test_features=test_features,
        train_x=train_x,
        test_x=test_x,
        train_times=train_times,
        test_times=test_times,
        train_labels_true=train_labels_true,
        test_labels_true=test_labels_true,
        seed=seed,
        n_init=n_init,
        alphabet_size=alphabet_size,
    )
    transition_accuracy = (
        _modulo_transition_accuracy(
            labels_test,
            test_labels_true["joint"],
            test_x,
            test_times,
            modulo_n=modulo_n,
        )
        if modulo_n is not None
        else math.nan
    )
    return {
        "ablation": ablation,
        "value": int(value),
        **row,
        "transition_accuracy": transition_accuracy,
    }


def _plot_results(rows: list[dict[str, object]], hidden: np.ndarray, joint: np.ndarray, labels: dict[str, np.ndarray], outdir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    candidate_ks = sorted({int(row["k"]) for row in rows if str(row["method"]) != "trained_gru"})
    predictive_rows = [row for row in rows if str(row["method"]) == "predictive_signature"]
    if predictive_rows:
        plot_k = int(max(predictive_rows, key=lambda row: float(row["ari_joint"]))["k"])
    elif 12 in candidate_ks:
        plot_k = 12
    elif 2 in candidate_ks:
        plot_k = 2
    else:
        plot_k = candidate_ks[-1]
    k_rows = [row for row in rows if int(row["k"]) == plot_k]
    methods = [str(row["method"]) for row in k_rows]
    x = np.arange(len(methods), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    for offset, metric, label in [
        (-width, "ari_coarse", "coarse"),
        (0.0, "ari_fine", "fine"),
        (width, "ari_joint", "joint"),
    ]:
        ax.bar(x + offset, [float(row[metric]) for row in k_rows], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Adjusted Rand index")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "rnn_hidden_state_recovery_summary.png", dpi=220)
    fig.savefig(figures / "rnn_hidden_state_recovery_summary.pdf")
    plt.close(fig)

    coords = _pca2(hidden)
    take = np.linspace(0, hidden.shape[0] - 1, min(hidden.shape[0], 2500), dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True, sharey=True)
    panels = [
        ("true joint state", joint),
        ("hidden k-means", labels.get(f"hidden_kmeans_k{plot_k}")),
        ("predictive signature", labels.get(f"predictive_signature_k{plot_k}")),
    ]
    for ax, (title, colour) in zip(axes, panels):
        if colour is None:
            continue
        scatter = ax.scatter(
            coords[take, 0],
            coords[take, 1],
            c=np.asarray(colour)[take],
            s=4,
            cmap="tab20",
            linewidths=0,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "rnn_hidden_state_recovery_scatter.png", dpi=220)
    fig.savefig(figures / "rnn_hidden_state_recovery_scatter.pdf")
    plt.close(fig)


def run(config: ExperimentConfig, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generator = _make_process(config)
    train_sample = generator.sample(config.train_length, seed=config.seed)
    test_sample = generator.sample(config.test_length, seed=config.seed + 10_000)
    train_x = np.asarray(train_sample.x, dtype=int)
    test_x = np.asarray(test_sample.x, dtype=int)
    train_latent = np.asarray(train_sample.latent, dtype=int)
    test_latent = np.asarray(test_sample.latent, dtype=int)
    alphabet_size = _alphabet_size(train_x)

    predictor = GRUPredictor(
        alphabet_size,
        config.embedding_dim,
        config.hidden_dim,
        seed=config.seed,
    )
    training_history = predictor.train(
        train_x,
        epochs=config.epochs,
        batches_per_epoch=config.batches_per_epoch,
        batch_size=config.batch_size,
        unroll=config.unroll,
        learning_rate=config.learning_rate,
        seed=config.seed + 1,
    )
    train_hidden, train_next_probs = predictor.extract(train_x)
    test_hidden, test_next_probs = predictor.extract(test_x)

    train_times = np.arange(config.burn_in, train_hidden.shape[0], dtype=int)
    test_times = np.arange(config.burn_in, test_hidden.shape[0], dtype=int)
    train_true_all = _label_dict(generator, train_latent)
    test_true_all = _label_dict(generator, test_latent)
    train_true = {key: value[train_times] for key, value in train_true_all.items()}
    test_true = {key: value[test_times] for key, value in test_true_all.items()}

    train_features = {
        "hidden_kmeans": train_hidden[train_times],
        "next_symbol_kmeans": train_next_probs[train_times],
        "predictive_signature": predictor.rollout_signatures(
            train_hidden[train_times],
            horizon=config.rollout_horizon,
        ),
        "history_kmeans": _one_hot_history(
            train_x,
            train_times,
            context_len=config.context_len,
            alphabet_size=alphabet_size,
        ),
    }
    test_features = {
        "hidden_kmeans": test_hidden[test_times],
        "next_symbol_kmeans": test_next_probs[test_times],
        "predictive_signature": predictor.rollout_signatures(
            test_hidden[test_times],
            horizon=config.rollout_horizon,
        ),
        "history_kmeans": _one_hot_history(
            test_x,
            test_times,
            context_len=config.context_len,
            alphabet_size=alphabet_size,
        ),
    }

    rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    label_payload: dict[str, np.ndarray] = {
        "test_coarse": test_true["coarse"],
        "test_fine": test_true["fine"],
        "test_joint": test_true["joint"],
    }
    for method in ("history_kmeans", "hidden_kmeans", "next_symbol_kmeans", "predictive_signature"):
        for k in config.cluster_ks:
            row, _, labels_test, _ = _evaluate_cluster_method(
                method=method,
                k=int(k),
                train_features=train_features[method],
                test_features=test_features[method],
                train_x=train_x,
                test_x=test_x,
                train_times=train_times,
                test_times=test_times,
                train_labels_true=train_true,
                test_labels_true=test_true,
                seed=config.seed + int(k),
                n_init=config.kmeans_n_init,
                alphabet_size=alphabet_size,
            )
            rows.append(row)
            label_payload[f"{method}_k{k}"] = labels_test

    ablation_k = config.modulo_n if config.process == "modulo" else int(config.cluster_ks[0])
    modulo_n = config.modulo_n if config.process == "modulo" else None
    for horizon in config.ablation_horizons:
        ablation_rows.append(
            _ablation_row(
                ablation="predictive_horizon",
                value=int(horizon),
                method="predictive_signature",
                k=ablation_k,
                train_features=predictor.rollout_signatures(
                    train_hidden[train_times],
                    horizon=int(horizon),
                ),
                test_features=predictor.rollout_signatures(
                    test_hidden[test_times],
                    horizon=int(horizon),
                ),
                train_x=train_x,
                test_x=test_x,
                train_times=train_times,
                test_times=test_times,
                train_labels_true=train_true,
                test_labels_true=test_true,
                seed=config.seed + 17_000 + int(horizon),
                n_init=config.kmeans_n_init,
                alphabet_size=alphabet_size,
                modulo_n=modulo_n,
            )
        )
    for context_len in config.ablation_context_lens:
        ablation_rows.append(
            _ablation_row(
                ablation="history_length",
                value=int(context_len),
                method="history_kmeans",
                k=ablation_k,
                train_features=_one_hot_history(
                    train_x,
                    train_times,
                    context_len=int(context_len),
                    alphabet_size=alphabet_size,
                ),
                test_features=_one_hot_history(
                    test_x,
                    test_times,
                    context_len=int(context_len),
                    alphabet_size=alphabet_size,
                ),
                train_x=train_x,
                test_x=test_x,
                train_times=train_times,
                test_times=test_times,
                train_labels_true=train_true,
                test_labels_true=test_true,
                seed=config.seed + 23_000 + int(context_len),
                n_init=config.kmeans_n_init,
                alphabet_size=alphabet_size,
                modulo_n=modulo_n,
            )
        )

    model_nll = -float(
        np.mean(
            np.log(
                np.maximum(
                    test_next_probs[test_times, test_x[test_times + 1]],
                    1e-12,
                )
            )
        )
    )
    rows.append(
        {
            "method": "trained_gru",
            "k": 0,
            "n_states": config.hidden_dim,
            "ari_coarse": math.nan,
            "ari_fine": math.nan,
            "ari_joint": math.nan,
            "test_logloss": model_nll,
            "unifilarity": math.nan,
            "branch_entropy": math.nan,
        }
    )

    fieldnames = [
        "method",
        "k",
        "n_states",
        "ari_coarse",
        "ari_fine",
        "ari_joint",
        "test_logloss",
        "unifilarity",
        "branch_entropy",
    ]
    save_csv(outdir / "recovery.csv", rows, fieldnames=fieldnames)
    if ablation_rows:
        save_csv(
            outdir / "ablation.csv",
            ablation_rows,
            fieldnames=[
                "ablation",
                "value",
                "method",
                "k",
                "n_states",
                "ari_coarse",
                "ari_fine",
                "ari_joint",
                "test_logloss",
                "unifilarity",
                "branch_entropy",
                "transition_accuracy",
            ],
        )
    save_csv(
        outdir / "training_history.csv",
        training_history,
        fieldnames=["epoch", "train_nll", "elapsed_s"],
    )
    save_json(outdir / "config.json", config)
    np.savez(
        outdir / "labels_and_features.npz",
        test_x=test_x,
        test_times=test_times,
        test_hidden=test_hidden[test_times],
        test_next_probs=test_next_probs[test_times],
        **label_payload,
    )
    if config.make_plots:
        _plot_results(
            rows,
            test_hidden[test_times],
            test_true["joint"],
            label_payload,
            outdir,
        )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("src/results/rnn_hidden_state_recovery"))
    parser.add_argument("--process", choices=["hierarchical", "even", "modulo"], default="hierarchical")
    parser.add_argument("--train-length", type=int, default=30_000)
    parser.add_argument("--test-length", type=int, default=12_000)
    parser.add_argument("--emission-noise", type=float, default=0.08)
    parser.add_argument("--even-p", type=float, default=0.7)
    parser.add_argument("--modulo-n", type=int, default=4)
    parser.add_argument("--modulo-reset-prob", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--unroll", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batches-per-epoch", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=3)
    parser.add_argument("--cluster-ks", nargs="+", type=int, default=[3, 4, 6, 12])
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--context-len", type=int, default=3)
    parser.add_argument("--ablation-horizons", nargs="*", type=int, default=[])
    parser.add_argument("--ablation-context-lens", nargs="*", type=int, default=[])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.train_length <= args.unroll + 1:
        raise ValueError("--train-length must exceed --unroll + 1.")
    if args.test_length <= args.burn_in + 2:
        raise ValueError("--test-length must exceed --burn-in + 2.")
    if args.rollout_horizon < 1:
        raise ValueError("--rollout-horizon must be positive.")
    if args.context_len < 1:
        raise ValueError("--context-len must be positive.")
    if any(horizon < 1 for horizon in args.ablation_horizons):
        raise ValueError("--ablation-horizons must contain positive values.")
    if any(context_len < 1 for context_len in args.ablation_context_lens):
        raise ValueError("--ablation-context-lens must contain positive values.")
    max_context_len = max([int(args.context_len), *(int(v) for v in args.ablation_context_lens)])
    if args.burn_in < max_context_len - 1:
        raise ValueError("--burn-in must be at least max history length - 1.")

    config = ExperimentConfig(
        process=str(args.process),
        train_length=int(args.train_length),
        test_length=int(args.test_length),
        emission_noise=float(args.emission_noise),
        even_p=float(args.even_p),
        modulo_n=int(args.modulo_n),
        modulo_reset_prob=float(args.modulo_reset_prob),
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        embedding_dim=int(args.embedding_dim),
        unroll=int(args.unroll),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        batches_per_epoch=int(args.batches_per_epoch),
        learning_rate=float(args.learning_rate),
        burn_in=int(args.burn_in),
        rollout_horizon=int(args.rollout_horizon),
        cluster_ks=tuple(int(k) for k in args.cluster_ks),
        kmeans_n_init=int(args.kmeans_n_init),
        context_len=int(args.context_len),
        ablation_horizons=tuple(int(v) for v in args.ablation_horizons),
        ablation_context_lens=tuple(int(v) for v in args.ablation_context_lens),
        make_plots=not bool(args.no_plots),
    )
    run(config, args.outdir)


if __name__ == "__main__":
    main()
