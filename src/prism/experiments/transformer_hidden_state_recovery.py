from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prism.experiments.rnn_hidden_state_recovery import (
    ModuloRenewalProcess,
    _kmeans_best,
)
from prism.reconstruction.kalman_iss import _adjusted_rand_index
from prism.utils.io import save_csv, save_json


def _kmeans_clusters(features, k, n_init, seed):
    labels, _ = _kmeans_best(features, k=k, seed=seed, n_init=n_init)
    return labels


def _ari(truth, labels):
    return _adjusted_rand_index(truth, labels)


@dataclass(frozen=True)
class TransformerConfig:
    # Process
    modulo_n: int
    modulo_reset_prob: float
    train_length: int
    test_length: int
    burn_in: int
    # Architecture
    d_model: int
    nhead: int
    num_layers: int
    ffn_dim: int
    max_context: int
    dropout: float
    # Training
    seed: int
    epochs: int
    batches_per_epoch: int
    batch_size: int
    unroll: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    # Readout
    rollout_horizon: int
    cluster_ks: tuple[int, ...]
    kmeans_n_init: int
    raw_history_lens: tuple[int, ...]
    ablation_horizons: tuple[int, ...]
    

def _torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "transformer_hidden_state_recovery requires torch. "
            "Use the project venv or install requirements.txt."
        ) from exc
    return torch


class _CausalTransformerNextSymbolModel:
    def __init__(
        self,
        alphabet_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ffn_dim: int,
        max_context: int,
        dropout: float,
    ) -> None:
        torch = _torch()
        nn = torch.nn
        self.alphabet_size = int(alphabet_size)
        self.d_model = int(d_model)
        self.max_context = int(max_context)

        self.embedding = nn.Embedding(alphabet_size, d_model)
        self.pos = self._sinusoidal_positional(max_context, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.readout = nn.Linear(d_model, alphabet_size)

    @staticmethod
    def _sinusoidal_positional(max_context: int, d_model: int):
        torch = _torch()
        pe = torch.zeros(max_context, d_model)
        position = torch.arange(0, max_context, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def parameters(self):
        return (
            list(self.embedding.parameters())
            + list(self.transformer.parameters())
            + list(self.readout.parameters())
        )

    def train(self) -> None:
        self.embedding.train()
        self.transformer.train()
        self.readout.train()

    def eval(self) -> None:
        self.embedding.eval()
        self.transformer.eval()
        self.readout.eval()

    def to(self, device):
        self.embedding = self.embedding.to(device)
        self.pos = self.pos.to(device)
        self.transformer = self.transformer.to(device)
        self.readout = self.readout.to(device)
        return self

    def _causal_mask(self, length: int, device):
        torch = _torch()
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def __call__(self, x):
        torch = _torch()
        B, L = x.shape
        if L > self.max_context:
            raise ValueError(f"input length {L} exceeds max_context {self.max_context}")
        device = x.device
        embed = self.embedding(x) + self.pos[:L].unsqueeze(0)
        mask = self._causal_mask(L, device=device)
        hidden = self.transformer(embed, mask=mask)
        logits = self.readout(hidden)
        return logits, hidden

class TransformerPredictor:
    def __init__(self, alphabet_size: int, cfg: TransformerConfig) -> None:
        torch = _torch()
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.torch = torch
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _CausalTransformerNextSymbolModel(
            alphabet_size=alphabet_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            ffn_dim=cfg.ffn_dim,
            max_context=cfg.max_context,
            dropout=cfg.dropout,
        ).to(self.device)
        self.alphabet_size = int(alphabet_size)

    def train(self, x: np.ndarray) -> list[dict[str, float]]:
        torch = self.torch
        cfg = self.cfg
        if x.shape[0] <= cfg.unroll + 1:
            raise ValueError("Training sequence is shorter than the unroll length.")

        rng = np.random.default_rng(cfg.seed)
        optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        history: list[dict[str, float]] = []
        max_start = x.shape[0] - cfg.unroll - 1

        self.model.train()
        started = time.perf_counter()
        global_step = 0
        for epoch in range(cfg.epochs):
            losses: list[float] = []
            for _ in range(cfg.batches_per_epoch):
                # Linear LR warmup over the first cfg.warmup_steps steps,
                # then constant at cfg.learning_rate. With warmup_steps=0
                if cfg.warmup_steps > 0 and global_step < cfg.warmup_steps:
                    lr_scale = (global_step + 1) / cfg.warmup_steps
                    for pg in optimiser.param_groups:
                        pg["lr"] = cfg.learning_rate * lr_scale

                starts = rng.integers(0, max_start + 1, size=cfg.batch_size)
                inputs = np.stack([x[s : s + cfg.unroll] for s in starts], axis=0)
                targets = np.stack(
                    [x[s + 1 : s + cfg.unroll + 1] for s in starts], axis=0
                )
                xt = torch.as_tensor(inputs, dtype=torch.long, device=self.device)
                yt = torch.as_tensor(targets, dtype=torch.long, device=self.device)
                logits, _ = self.model(xt)
                loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]), yt.reshape(-1)
                )
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimiser.step()
                losses.append(float(loss.detach().cpu().item()))
                global_step += 1
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
        cfg = self.cfg
        self.model.eval()
        L = x.shape[0]
        win = min(cfg.max_context, max(cfg.unroll, 32))
        hidden_out: list[np.ndarray] = []
        probs_out: list[np.ndarray] = []
        with torch.no_grad():
            for end in range(1, L):
                start = max(0, end - win)
                ctx = x[start:end]
                xt = torch.as_tensor(
                    ctx[None, :], dtype=torch.long, device=self.device
                )
                logits, hidden = self.model(xt)
                hidden_out.append(hidden[0, -1].cpu().numpy())
                probs_out.append(
                    torch.softmax(logits[0, -1], dim=-1).cpu().numpy()
                )
        return (
            np.asarray(hidden_out, dtype=float),
            np.asarray(probs_out, dtype=float),
        )

    def rollout_signatures(
        self,
        contexts: list[np.ndarray],
        *,
        horizon: int,
    ) -> np.ndarray:
        torch = self.torch
        A = self.alphabet_size
        N = len(contexts)
        self.model.eval()
        sig = np.zeros((N, horizon, A), dtype=float)
        with torch.no_grad():
            for i, ctx in enumerate(contexts):
                self._rollout_recursive(
                    ctx=ctx,
                    weight=1.0,
                    lag=0,
                    horizon=horizon,
                    out=sig[i],
                )
        return sig.reshape(N, horizon * A)

    def _rollout_recursive(
        self,
        ctx: np.ndarray,
        weight: float,
        lag: int,
        horizon: int,
        out: np.ndarray,
    ) -> None:
        torch = self.torch
        if lag >= horizon:
            return
        cfg = self.cfg
        win = min(cfg.max_context, max(cfg.unroll, 32))
        view = ctx[-win:]
        xt = torch.as_tensor(view[None, :], dtype=torch.long, device=self.device)
        logits, _ = self.model(xt)
        probs = torch.softmax(logits[0, -1], dim=-1).cpu().numpy()
        out[lag] += weight * probs
        if lag == horizon - 1:
            return
        for sym in range(self.alphabet_size):
            extended = np.concatenate([ctx, np.asarray([sym], dtype=ctx.dtype)])
            self._rollout_recursive(
                ctx=extended,
                weight=weight * float(probs[sym]),
                lag=lag + 1,
                horizon=horizon,
                out=out,
            )


def _gen_modulo(
    n_states: int, reset_prob: float, length: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    proc = ModuloRenewalProcess(n_states=n_states, reset_prob=reset_prob)
    sample = proc.sample(length=length, seed=seed)
    return (
        np.asarray(sample.x, dtype=np.int64),
        np.asarray(sample.latent, dtype=np.int64),
    )


def _raw_history_features(
    x: np.ndarray, *, history_len: int, alphabet_size: int
) -> np.ndarray:
    L = x.shape[0]
    out = np.zeros((L - 1, history_len * alphabet_size), dtype=float)
    for end in range(1, L):
        start = max(0, end - history_len)
        suffix = x[start:end]
        for offset, sym in enumerate(suffix):
            out[end - 1, offset * alphabet_size + int(sym)] = 1.0
    return out


def _evaluate_clusters(
    features: np.ndarray,
    true_states: np.ndarray,
    *,
    ks: tuple[int, ...],
    n_init: int,
    seed: int,
    method: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for k in ks:
        labels = _kmeans_clusters(features, k=k, n_init=n_init, seed=seed)
        ari = _ari(true_states, labels)
        rows.append(
            {
                "method": method,
                "k": int(k),
                "ari": float(ari),
            }
        )
    return rows


def run(cfg: TransformerConfig, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    A = 2
    
    # Training and test data with the same generator seed as the GRU run
    x_train, _ = _gen_modulo(
        cfg.modulo_n, cfg.modulo_reset_prob, cfg.train_length, cfg.seed
    )
    x_test, s_test = _gen_modulo(
        cfg.modulo_n, cfg.modulo_reset_prob, cfg.test_length, cfg.seed + 10_000
    )

    pred = TransformerPredictor(alphabet_size=A, cfg=cfg)
    history = pred.train(x_train)

    # Hidden states and one-step probs over the test sequence
    hidden, one_step = pred.extract(x_test)

    # Predictive signatures: for each test position, the context is x_test[:t+1]
    contexts = [x_test[: end] for end in range(1, x_test.shape[0])]
    sig = pred.rollout_signatures(contexts, horizon=cfg.rollout_horizon)
    
    truth = s_test[cfg.burn_in : len(hidden) + 1]
    hidden_eval = hidden[cfg.burn_in - 1 : len(truth) + cfg.burn_in - 1]
    one_step_eval = one_step[cfg.burn_in - 1 : len(truth) + cfg.burn_in - 1]
    sig_eval = sig[cfg.burn_in - 1 : len(truth) + cfg.burn_in - 1]

    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        _evaluate_clusters(
            sig_eval,
            truth,
            ks=cfg.cluster_ks,
            n_init=cfg.kmeans_n_init,
            seed=cfg.seed,
            method="transformer_predictive_signature",
        )
    )
    rows.extend(
        _evaluate_clusters(
            hidden_eval,
            truth,
            ks=cfg.cluster_ks,
            n_init=cfg.kmeans_n_init,
            seed=cfg.seed,
            method="transformer_hidden",
        )
    )
    rows.extend(
        _evaluate_clusters(
            one_step_eval,
            truth,
            ks=cfg.cluster_ks,
            n_init=cfg.kmeans_n_init,
            seed=cfg.seed,
            method="transformer_one_step",
        )
    )
    for hl in cfg.raw_history_lens:
        feats = _raw_history_features(
            x_test[cfg.burn_in - hl : len(truth) + cfg.burn_in],
            history_len=hl,
            alphabet_size=A,
        )
        feats = feats[: len(truth)]
        rows.extend(
            _evaluate_clusters(
                feats,
                truth,
                ks=cfg.cluster_ks,
                n_init=cfg.kmeans_n_init,
                seed=cfg.seed,
                method=f"raw_history_len_{hl}",
            )
        )

    # Rollout-horizon ablation: clusters the signature at varying H
    ablation_rows: list[dict[str, float | int | str]] = []
    for H in cfg.ablation_horizons:
        if H == cfg.rollout_horizon:
            sig_H = sig_eval
        else:
            sig_H = pred.rollout_signatures(contexts, horizon=H)
            sig_H = sig_H[cfg.burn_in - 1 : len(truth) + cfg.burn_in - 1]
        labels = _kmeans_clusters(
            sig_H, k=cfg.modulo_n, n_init=cfg.kmeans_n_init, seed=cfg.seed
        )
        ari = _ari(truth, labels)
        ablation_rows.append(
            {
                "ablation": "rollout_horizon",
                "value": int(H),
                "ari": float(ari),
            }
        )

    save_csv(outdir / "recovery.csv", rows)
    if ablation_rows:
        save_csv(outdir / "ablations.csv", ablation_rows)
    save_json(
        outdir / "recovery.json",
        {
            "config": cfg.__dict__,
            "final_train_nll": float(history[-1]["train_nll"]),
            "wall_time_s": float(history[-1]["elapsed_s"]),
            "rows": rows,
            "ablations": ablation_rows,
        },
    )

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--outdir", type=Path, default=Path("src/results/transformer_hidden_state_recovery")
    )
    p.add_argument("--seed", type=int, default=0)
    # Process
    p.add_argument("--modulo-n", type=int, default=5)
    p.add_argument("--modulo-reset-prob", type=float, default=0.35)
    p.add_argument("--train-length", type=int, default=30_000)
    p.add_argument("--test-length", type=int, default=12_000)
    p.add_argument("--burn-in", type=int, default=200)
    # Architecture
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=128)
    p.add_argument("--max-context", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    # Training
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batches-per-epoch", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--unroll", type=int, default=80)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=0)
    # Readout
    p.add_argument("--rollout-horizon", type=int, default=5)
    p.add_argument("--cluster-ks", nargs="+", type=int, default=[3, 4, 5, 6, 8])
    p.add_argument("--kmeans-n-init", type=int, default=20)
    p.add_argument(
        "--raw-history-lens", nargs="*", type=int, default=[3, 10, 30, 50]
    )
    p.add_argument(
        "--ablation-horizons", nargs="*", type=int, default=[1, 2, 3, 4, 5]
    )
    args = p.parse_args()

    cfg = TransformerConfig(
        modulo_n=args.modulo_n,
        modulo_reset_prob=args.modulo_reset_prob,
        train_length=args.train_length,
        test_length=args.test_length,
        burn_in=args.burn_in,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        max_context=args.max_context,
        dropout=args.dropout,
        seed=args.seed,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        unroll=args.unroll,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        rollout_horizon=args.rollout_horizon,
        cluster_ks=tuple(args.cluster_ks),
        kmeans_n_init=args.kmeans_n_init,
        raw_history_lens=tuple(args.raw_history_lens),
        ablation_horizons=tuple(args.ablation_horizons),
    )
    run(cfg, args.outdir)


if __name__ == "__main__":
    main()
