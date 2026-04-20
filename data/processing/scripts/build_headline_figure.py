"""Publication-grade headline figure for the central pre-stim VAR predictive-fit
detection signal:

  Panel A — Forest plot of trial-level LME beta on sdt across the 15 region x
            window cells (95% CI, Holm-significance markers).
  Panel B — Leave-one-subject-out distribution: drop each subject in turn,
            show the resulting beta and z (rug + median band).
  Panel C — Within-subject permutation null distribution against the observed
            central 125-375 beta.
  Panel D — Per-subject Spearman partial correlation r(pred_fit, hit | stimamp)
            and r(pred_fit, conf | stimamp), paired strip with subject lines.

All four panels combine into a single 2x2 figure suitable for a journal headline."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SWEEP_TABLE = Path(
    "/Users/aniketgupta/Desktop/Imperial/FourthYear/FYP/PRISM/data/results_baseline/"
    "region_sliding_baseline300ms_controls_focus_q4/"
    "summary_mechanistic_dissociation_sweep/mechanistic_sweep_table.csv"
)
LOSO_TABLE = Path(
    "/Users/aniketgupta/Desktop/Imperial/FourthYear/FYP/PRISM/data/results_baseline/"
    "region_sliding_baseline300ms_controls_focus_q4/"
    "summary_mechanistic_robustness_central/loso_table.csv"
)
PERM_TABLE = Path(
    "/Users/aniketgupta/Desktop/Imperial/FourthYear/FYP/PRISM/data/results_baseline/"
    "region_sliding_baseline300ms_controls_focus_q4/"
    "summary_mechanistic_robustness_central/permutation_null_betas.csv"
)
PER_SUBJECT_TABLE = Path(
    "/Users/aniketgupta/Desktop/Imperial/FourthYear/FYP/PRISM/data/results_baseline/"
    "region_sliding_baseline300ms_controls_focus_q4/"
    "summary_mechanistic_dissociation_central/per_subject_spearman_partial.csv"
)
OBSERVED_BETA = 0.04607139563581471
OUT_PNG = Path(
    "/Users/aniketgupta/Desktop/Imperial/FourthYear/FYP/PRISM/data/results_baseline/"
    "region_sliding_baseline300ms_controls_focus_q4/"
    "summary_mechanistic_dissociation_sweep/headline_figure.png"
)
OUT_PDF = OUT_PNG.with_suffix(".pdf")

REGION_ORDER = ("central", "frontal", "parietal", "occipital", "temporal")
WINDOW_ORDER = ((0.0, 250.0), (125.0, 375.0), (250.0, 500.0))


def panel_a_forest(ax: plt.Axes, sweep: pd.DataFrame) -> None:
    cells = []
    for region in REGION_ORDER:
        for start, end in WINDOW_ORDER:
            row = sweep[
                (sweep["region"] == region)
                & (sweep["window_start_ms"] == start)
                & (sweep["window_end_ms"] == end)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            cells.append(
                {
                    "label": f"{region} {int(start)}-{int(end)}ms",
                    "beta": row["lme_det_beta"],
                    "ci_lo": row["lme_det_ci95_low"],
                    "ci_hi": row["lme_det_ci95_high"],
                    "holm_p": row["lme_det_p_holm"],
                    "z": row["lme_det_z"],
                }
            )
    cells_df = pd.DataFrame(cells)
    cells_df = cells_df.iloc[::-1].reset_index(drop=True)  # flip for top-down

    y = np.arange(len(cells_df))
    colours = []
    for p in cells_df["holm_p"]:
        if p < 1e-10:
            colours.append("#0c2a4d")
        elif p < 0.001:
            colours.append("#155f9e")
        elif p < 0.05:
            colours.append("#7fb3df")
        else:
            colours.append("#bababa")

    err_lo = cells_df["beta"] - cells_df["ci_lo"]
    err_hi = cells_df["ci_hi"] - cells_df["beta"]
    ax.errorbar(
        cells_df["beta"], y,
        xerr=[err_lo.values, err_hi.values],
        fmt="o", color="#222", ecolor="#888", capsize=2.5, markersize=4, lw=1.0
    )
    for xi, yi, ci in zip(cells_df["beta"], y, colours):
        ax.plot(xi, yi, marker="o", color=ci, markersize=7, zorder=5,
                markeredgecolor="black", markeredgewidth=0.4)

    ax.axvline(0.0, color="#444", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(cells_df["label"], fontsize=8)
    ax.set_xlabel(r"LME $\beta$ on hit-vs-miss (95% CI)", fontsize=9)
    ax.set_title("A — Trial-level detection effect across regions × windows",
                 fontsize=10, loc="left")
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.2)
    ax.set_xlim(-0.005, 0.07)

    legend_h = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#0c2a4d",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.4,
                   label="Holm p < 1e-10"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#155f9e",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.4,
                   label="Holm p < 0.001"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#7fb3df",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.4,
                   label="Holm p < 0.05"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#bababa",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.4,
                   label="ns"),
    ]
    ax.legend(handles=legend_h, fontsize=7, loc="lower right", frameon=False)


def panel_b_loso(ax: plt.Axes, loso: pd.DataFrame) -> None:
    sorted_df = loso.sort_values("beta").reset_index(drop=True)
    y = np.arange(len(sorted_df))
    ax.barh(y, sorted_df["beta"], color="#155f9e", alpha=0.85,
            edgecolor="black", linewidth=0.3)
    ax.axvline(OBSERVED_BETA, color="#d62728", lw=1.2, ls="--",
               label=f"Full-sample β = {OBSERVED_BETA:+.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_df["dropped_subject"], fontsize=7)
    ax.set_xlabel(r"LOSO $\beta$ on hit-vs-miss", fontsize=9)
    ax.set_title("B — Leave-one-subject-out (drop each subject, refit)",
                 fontsize=10, loc="left")
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.2)
    ax.set_xlim(0.0, max(sorted_df["beta"].max(), OBSERVED_BETA) * 1.1)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    z_min = sorted_df["z"].min()
    z_max = sorted_df["z"].max()
    ax.text(0.02, 0.98,
            f"all 18 fits z ∈ [{z_min:.1f}, {z_max:.1f}]\n"
            f"max p = {sorted_df['p'].max():.1e}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="#aaa", lw=0.5, pad=2))


def panel_c_perm(ax: plt.Axes, perm: pd.DataFrame) -> None:
    null_betas = perm["null_beta"].dropna().values
    n_perm = len(null_betas)
    bins = np.linspace(min(null_betas.min(), -0.02), OBSERVED_BETA * 1.05, 60)
    ax.hist(null_betas, bins=bins, color="#9aa0a6", edgecolor="white",
            linewidth=0.4, label=f"Within-subject permutation null\n(n={n_perm})")
    ax.axvline(OBSERVED_BETA, color="#d62728", lw=1.5,
               label=f"Observed β = {OBSERVED_BETA:+.3f}")
    ax.set_xlabel(r"$\beta$ on hit-vs-miss", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title("C — Permutation null vs observed effect",
                 fontsize=10, loc="left")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.2)
    n_extreme = int(np.sum(np.abs(null_betas) >= abs(OBSERVED_BETA)))
    empirical_p = (n_extreme + 1) / (n_perm + 1)
    null_z = (OBSERVED_BETA - null_betas.mean()) / (null_betas.std() + 1e-12)
    ax.text(0.55, 0.85,
            f"{n_extreme}/{n_perm} null betas exceed observed\n"
            f"empirical p = {empirical_p:.4f}\n"
            f"null mean ± SD: {null_betas.mean():+.4f} ± {null_betas.std():.4f}\n"
            f"observed z (vs null) = {null_z:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="#aaa", lw=0.5, pad=3))
    ax.legend(fontsize=7, loc="upper left", frameon=False)


def panel_d_per_subject(ax: plt.Axes, per_sub: pd.DataFrame) -> None:
    df = per_sub.copy()
    df = df.dropna(
        subset=[
            "spearman_partial_pred_fit_vs_detection",
            "spearman_partial_pred_fit_vs_confidence",
        ]
    )
    rng = np.random.default_rng(7)
    jit_det = rng.uniform(-0.05, 0.05, len(df))
    jit_conf = rng.uniform(-0.05, 0.05, len(df))

    for d, c, jd, jc in zip(
        df["spearman_partial_pred_fit_vs_detection"],
        df["spearman_partial_pred_fit_vs_confidence"],
        jit_det, jit_conf,
    ):
        ax.plot([0 + jd, 1 + jc], [d, c], color="#bbb", lw=0.6, alpha=0.7, zorder=1)
    ax.scatter(np.zeros(len(df)) + jit_det,
               df["spearman_partial_pred_fit_vs_detection"],
               color="#155f9e", s=40, edgecolor="black", linewidth=0.4,
               label="r(pred_fit, hit | stimamp)", zorder=3)
    ax.scatter(np.ones(len(df)) + jit_conf,
               df["spearman_partial_pred_fit_vs_confidence"],
               color="#e36414", s=40, edgecolor="black", linewidth=0.4,
               label="r(pred_fit, confidence | stimamp)", zorder=3)
    ax.axhline(0, color="#444", lw=0.6, ls="--")

    det_mean = df["spearman_partial_pred_fit_vs_detection"].mean()
    conf_mean = df["spearman_partial_pred_fit_vs_confidence"].mean()
    ax.plot([-0.18, 0.18], [det_mean, det_mean], color="#0c2a4d", lw=2)
    ax.plot([0.82, 1.18], [conf_mean, conf_mean], color="#a04a08", lw=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["detection", "confidence (hits only)"], fontsize=9)
    ax.set_ylabel("per-subject Spearman partial r", fontsize=9)
    ax.set_title("D — Per-subject continuous dissociation",
                 fontsize=10, loc="left")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_xlim(-0.45, 1.45)
    ax.text(0.02, 0.98,
            f"detection mean r = {det_mean:+.3f}, t-test p = 7.3e-5\n"
            f"confidence mean r = {conf_mean:+.3f}, t-test p = 0.0087\n"
            f"paired (det − conf) p = 0.105",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="#aaa", lw=0.5, pad=3))
    ax.legend(fontsize=7, loc="lower left", frameon=False)


def main() -> int:
    sweep = pd.read_csv(SWEEP_TABLE)
    loso = pd.read_csv(LOSO_TABLE)
    perm = pd.read_csv(PERM_TABLE)
    per_sub = pd.read_csv(PER_SUBJECT_TABLE)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.5),
                             gridspec_kw={"hspace": 0.36, "wspace": 0.32})

    panel_a_forest(axes[0, 0], sweep)
    panel_b_loso(axes[0, 1], loso)
    panel_c_perm(axes[1, 0], perm)
    panel_d_per_subject(axes[1, 1], per_sub)

    fig.suptitle(
        "Pre-stimulus multivariate dynamical structure encodes a graded "
        "perceptual-state signal at central scalp",
        fontsize=12, y=0.995, x=0.51
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
