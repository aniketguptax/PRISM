"""Aggregate EEG CE 2.0 metrics from per-trial label archives."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prism.analysis.causal_emergence import (
    cp_path_from_label_chain,
    delta_cp,
    emergent_complexity,
    total_ce,
)


def _load_trial_npz(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def _label_chain_from_npz(
    data: dict,
    eps_values: list[float] | None = None,
) -> list[np.ndarray]:
    entries: list[tuple[int, np.ndarray]] = []
    for key, values in data.items():
        if not key.startswith("labels_eps"):
            continue
        try:
            eps = float(key[len("labels_eps") :])
        except ValueError:
            continue
        if eps_values is not None and eps not in eps_values:
            continue
        labels = np.asarray(values, dtype=int)
        n_states = int(labels.max()) + 1
        entries.append((n_states, labels))
    if not entries:
        return []

    entries.sort(key=lambda item: -item[0])
    chain: list[np.ndarray] = []
    seen_sizes: set[int] = set()
    for n_states, labels in entries:
        if n_states in seen_sizes:
            continue
        seen_sizes.add(n_states)
        chain.append(labels)
    return chain


def _trial_ce_metrics(chain: list[np.ndarray]) -> dict[str, float | int]:
    if len(chain) < 2:
        finest = int(chain[0].max()) + 1 if chain else 0
        coarsest = int(chain[-1].max()) + 1 if chain else 0
        return {
            "n_rungs": len(chain),
            "n_states_finest": finest,
            "n_states_coarsest": coarsest,
            "cp_finest": float("nan"),
            "cp_coarsest": float("nan"),
            "total_ce": float("nan"),
            "emergent_complexity": float("nan"),
        }

    rungs = cp_path_from_label_chain(chain, use_observed_distribution=True)
    deltas = delta_cp(rungs)
    return {
        "n_rungs": len(rungs),
        "n_states_finest": rungs[0].n_states,
        "n_states_coarsest": rungs[-1].n_states,
        "cp_finest": float(rungs[0].cp),
        "cp_coarsest": float(rungs[-1].cp),
        "total_ce": float(total_ce(rungs)),
        "emergent_complexity": float(emergent_complexity(deltas, normalise=True)),
    }


def _scalar(data: dict, key: str, default=float("nan")):
    if key not in data:
        return default
    value = data[key]
    return value.item() if hasattr(value, "item") else value


def _plot_by_region(trials: pd.DataFrame, figdir: Path) -> None:
    metrics = [
        ("total_ce", "Total CE"),
        ("emergent_complexity", "Emergent complexity"),
    ]
    for region in sorted(trials["region_name"].unique()):
        region_df = trials[trials["region_name"] == region]
        windows = sorted(region_df["window_name"].unique())
        fig, axes = plt.subplots(
            len(metrics),
            len(windows),
            figsize=(3.5 * len(windows), 3.5 * len(metrics)),
            squeeze=False,
        )
        for col, window in enumerate(windows):
            window_df = region_df[region_df["window_name"] == window]
            for row, (metric, label) in enumerate(metrics):
                ax = axes[row][col]
                misses = window_df[window_df["hit"] == 0][metric].dropna()
                hits = window_df[window_df["hit"] == 1][metric].dropna()
                ax.boxplot(
                    [misses.values, hits.values],
                    labels=["miss", "hit"],
                    widths=0.5,
                    medianprops={"color": "black", "linewidth": 1.5},
                )
                if row == 0:
                    ax.set_title(window.replace("_", " "), fontsize=9)
                if col == 0:
                    ax.set_ylabel(label, fontsize=9)
                ax.grid(True, alpha=0.3)

        fig.suptitle(f"CE 2.0 - {region}", fontsize=10)
        fig.tight_layout()
        outpath = figdir / f"emergence_{region}.pdf"
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {outpath}")


def run(root: Path, outdir: Path, eps_values: list[float] | None = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    npz_paths = sorted(root.glob("trial_*.npz"))
    if not npz_paths:
        npz_paths = sorted(root.rglob("trial_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No trial_*.npz files found under {root}")

    rows: list[dict] = []
    for path in npz_paths:
        data = _load_trial_npz(path)
        chain = _label_chain_from_npz(data, eps_values=eps_values)
        row = {
            "subject": str(_scalar(data, "subject", "")),
            "trial_idx": int(_scalar(data, "trial_idx", -1)),
            "region_name": str(_scalar(data, "region_name", "")),
            "window_name": str(_scalar(data, "window_name", "")),
            "hit": int(_scalar(data, "hit", -1)),
            "confidence": float(_scalar(data, "confidence", float("nan"))),
            "stim_amp": float(_scalar(data, "stim_amp", float("nan"))),
        }
        row.update(_trial_ce_metrics(chain))
        rows.append(row)

    trials = pd.DataFrame(rows)
    trials.to_csv(outdir / "emergence_trials.csv", index=False)
    print(f"Wrote {len(trials)} trial rows -> {outdir / 'emergence_trials.csv'}")

    valid = trials.dropna(subset=["total_ce"])
    valid = valid[valid["hit"].isin([0, 1])]

    summary_rows: list[dict] = []
    for (region, window, hit), group in valid.groupby(["region_name", "window_name", "hit"]):
        for metric in ("total_ce", "emergent_complexity", "cp_coarsest", "cp_finest"):
            values = group[metric].dropna()
            if values.empty:
                continue
            summary_rows.append(
                {
                    "region_name": region,
                    "window_name": window,
                    "condition": "hit" if hit else "miss",
                    "metric": metric,
                    "n": int(len(group)),
                    "mean": float(values.mean()),
                    "sem": float(values.sem()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "emergence_summary.csv", index=False)
    print(f"Wrote summary -> {outdir / 'emergence_summary.csv'}")

    if not valid.empty:
        _plot_by_region(valid, figdir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate EEG CE 2.0 metrics from per-trial npz archives.")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing trial_*.npz files")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory, defaulting to root/emergence")
    parser.add_argument("--eps", type=float, nargs="*", default=None, help="Restrict to these eps_macro values")
    args = parser.parse_args()

    outdir = args.outdir if args.outdir is not None else args.root / "emergence"
    run(root=args.root, outdir=outdir, eps_values=args.eps)


if __name__ == "__main__":
    main()
