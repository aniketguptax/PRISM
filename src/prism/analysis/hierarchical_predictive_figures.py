"""Publication figures for the hierarchical predictive-state benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


COLORS = {
    "prism_predictive": "#1f6fb2",
    "history_kmeans": "#c7352f",
    "ink": "#2f3338",
    "muted": "#666d75",
    "grid": "#d8dce0",
    "box": "#f5f7fa",
    "box_alt": "#eef4fb",
}

METHOD_LABELS = {
    "prism_predictive": "PRISM predictive",
    "history_kmeans": "History k-means",
}


def _best_by_noise(df: pd.DataFrame, metric: str, *, maximise: bool) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for noise, sub in df.groupby("noise"):
        row: dict[str, float | str] = {"noise": float(noise)}
        for method in ("prism_predictive", "history_kmeans"):
            method_sub = sub[sub["method"] == method]
            means = (
                method_sub.groupby("method_param", as_index=False)[metric]
                .agg(["mean", "sem"])
                .reset_index()
            )
            idx = means["mean"].idxmax() if maximise else means["mean"].idxmin()
            best = means.loc[idx]
            row[f"{method}_param"] = float(best["method_param"])
            row[f"{method}_mean"] = float(best["mean"])
            row[f"{method}_sem"] = 0.0 if np.isnan(best["sem"]) else float(best["sem"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("noise")


def _write_summary_tables(df: pd.DataFrame, outdir: Path) -> None:
    joint = _best_by_noise(df, "ari_joint", maximise=True)
    nll = _best_by_noise(df, "test_logloss", maximise=False)

    summary = pd.DataFrame(
        {
            "noise": joint["noise"],
            "prism_best_eps_joint": joint["prism_predictive_param"],
            "prism_best_joint": joint["prism_predictive_mean"],
            "kmeans_best_k_joint": joint["history_kmeans_param"],
            "kmeans_best_joint": joint["history_kmeans_mean"],
            "joint_gain": joint["prism_predictive_mean"] - joint["history_kmeans_mean"],
            "prism_best_eps_nll": nll["prism_predictive_param"],
            "prism_best_nll": nll["prism_predictive_mean"],
            "kmeans_best_k_nll": nll["history_kmeans_param"],
            "kmeans_best_nll": nll["history_kmeans_mean"],
            "nll_gain": nll["history_kmeans_mean"] - nll["prism_predictive_mean"],
        }
    )
    summary.to_csv(outdir / "hierarchical_predictive_summary.csv", index=False)

    scale = (
        df[df["method"] == "prism_predictive"]
        .groupby(["noise", "method_param"], as_index=False)
        .agg(
            n_states=("n_states", "mean"),
            ari_joint=("ari_joint", "mean"),
            test_logloss=("test_logloss", "mean"),
            unifilarity=("unifilarity", "mean"),
            branch_entropy=("branch_entropy", "mean"),
        )
    )
    scale.to_csv(outdir / "hierarchical_predictive_scale_path.csv", index=False)


def _plot_generator_schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.03, 0.60, 0.24, 0.19, "Coarse\nregime $C_t$\n(slow)", COLORS["box_alt"]),
        (0.38, 0.60, 0.24, 0.19, "Fine\nphase $F_t$\n(fast)", COLORS["box"]),
        (0.73, 0.60, 0.24, 0.19, "Observed\nsymbol $Y_t$\n(noisy)", COLORS["box"]),
        (0.30, 0.22, 0.40, 0.17, "Future predictive\ndistribution\n$p(Y_{t+1:t+4}\\mid h_t)$", COLORS["box_alt"]),
    ]
    for x, y, w, h, label, facecolor in boxes:
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.012",
            facecolor="#f4f5f7",
            edgecolor=COLORS["ink"],
            linewidth=0.9,
            mutation_aspect=1,
        )
        rect.set_facecolor(facecolor)
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            linespacing=1.15,
            color=COLORS["ink"],
        )

    arrow = dict(arrowstyle="->", color=COLORS["ink"], linewidth=1.05, shrinkA=3, shrinkB=3)
    ax.annotate("", xy=(0.38, 0.695), xytext=(0.27, 0.695), arrowprops=arrow)
    ax.annotate("", xy=(0.73, 0.695), xytext=(0.62, 0.695), arrowprops=arrow)
    ax.annotate("", xy=(0.50, 0.40), xytext=(0.50, 0.60), arrowprops=arrow)
    ax.text(0.325, 0.805, "sets dynamics", ha="center", va="bottom", fontsize=5.3, color=COLORS["muted"])
    ax.text(0.675, 0.805, "emits", ha="center", va="bottom", fontsize=5.3, color=COLORS["muted"])
    ax.text(0.527, 0.50, "predicts", ha="left", va="center", fontsize=5.3, color=COLORS["muted"])
    ax.text(
        0.50,
        0.075,
        "The coarse regime is expressed through future dynamics,\nnot by immediate symbol frequency.",
        ha="center",
        va="center",
        fontsize=6.5,
        linespacing=1.18,
        color=COLORS["muted"],
    )


def _plot_best_curves(ax: plt.Axes, best: pd.DataFrame, metric: str, ylabel: str) -> None:
    for method in ("prism_predictive", "history_kmeans"):
        ax.errorbar(
            best["noise"],
            best[f"{method}_mean"],
            yerr=best[f"{method}_sem"],
            marker="o",
            markersize=4.2,
            linewidth=1.45,
            capsize=2.4,
            capthick=0.9,
            elinewidth=0.9,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Emission noise")
    ax.set_ylabel(ylabel)
    ax.set_xticks(best["noise"])
    ax.set_xticklabels([f"{value:.2f}" for value in best["noise"]])
    ax.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    if metric == "ari_joint":
        ax.set_ylim(0.0, 0.45)
    else:
        y_min = float(best[["prism_predictive_mean", "history_kmeans_mean"]].min().min())
        y_max = float(best[["prism_predictive_mean", "history_kmeans_mean"]].max().max())
        pad = max((y_max - y_min) * 0.12, 0.01)
        ax.set_ylim(y_min - pad, y_max + pad)


def _plot_scale_path(df: pd.DataFrame, ax_left: plt.Axes) -> None:
    selected = df[(df["method"] == "prism_predictive") & (df["noise"].isin([0.02, 0.08, 0.16]))]
    grouped = (
        selected.groupby(["noise", "method_param"], as_index=False)
        .agg(
            n_states=("n_states", "mean"),
            ari_joint=("ari_joint", "mean"),
            test_logloss=("test_logloss", "mean"),
        )
    )
    scale_colors = {0.02: "#5b1a7a", 0.08: "#157f7a", 0.16: "#d39b00"}
    noises = sorted(grouped["noise"].unique())
    for noise in noises:
        sub = grouped[grouped["noise"] == noise]
        ax_left.plot(
            sub["method_param"],
            sub["ari_joint"],
            marker="o",
            markersize=4.0,
            linewidth=1.45,
            color=scale_colors.get(float(noise), COLORS["prism_predictive"]),
            label=f"noise={noise:g}",
        )
    ax_left.set_xlabel("PRISM merge tolerance")
    ax_left.set_ylabel("Joint ARI")
    ax_left.set_xticks(sorted(grouped["method_param"].unique()))
    ax_left.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    ax_left.legend(fontsize=6.8, frameon=False, loc="lower left")

    ax_right = ax_left.twinx()
    state_means = grouped.groupby("method_param", as_index=False)["n_states"].mean()
    ax_right.plot(
        state_means["method_param"],
        state_means["n_states"],
        marker="s",
        linestyle="--",
        markersize=3.8,
        linewidth=1.05,
        color="#4b4f54",
        label="mean states",
    )
    ax_right.set_ylabel("Mean state count")
    ax_right.tick_params(axis="y", labelcolor="#4b4f54")


def make_figure(root: Path) -> Path:
    df = pd.read_csv(root / "recovery.csv")
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_tables(df, figures_dir)

    best_joint = _best_by_noise(df, "ari_joint", maximise=True)
    best_nll = _best_by_noise(df, "test_logloss", maximise=False)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "savefig.dpi": 350,
        }
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.4, 5.35),
        constrained_layout=True,
    )
    _plot_generator_schematic(axes[0, 0])
    axes[0, 0].set_title("A. Benchmark design", loc="left", fontweight="bold", pad=4)

    _plot_best_curves(axes[0, 1], best_joint, "ari_joint", "Best joint ARI")
    axes[0, 1].set_title("B. Hidden-state recovery", loc="left", fontweight="bold", pad=4)
    axes[0, 1].legend(frameon=False, loc="upper right", handlelength=1.8)

    _plot_best_curves(axes[1, 0], best_nll, "test_logloss", "Best held-out NLL")
    axes[1, 0].set_title("C. Predictive log-loss", loc="left", fontweight="bold", pad=4)

    _plot_scale_path(df, axes[1, 1])
    axes[1, 1].set_title("D. PRISM scale path", loc="left", fontweight="bold", pad=4)

    outpath = figures_dir / "hierarchical_predictive_main.png"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(figures_dir / "hierarchical_predictive_main.pdf", bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    outpath = make_figure(args.root)
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
