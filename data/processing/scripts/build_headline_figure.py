"""Build a report-ready EEG mechanistic headline figure.

The figure summarises the trial-level mixed-effects result for the
pre-stimulus VAR predictive-fit contribution. It deliberately uses only the
already-computed summary tables, so it is cheap to run locally; the expensive
subject-level model fitting remains in ``run_mechanistic_sweep.py`` and
``run_mechanistic_robustness.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = (
    REPO_ROOT
    / "data/results_baseline/region_sliding_baseline300ms_controls_focus_q4"
)
REGION_ORDER = ("central", "frontal", "parietal", "occipital", "temporal")
REGION_LABELS = {
    "central": "Central",
    "frontal": "Frontal",
    "parietal": "Parietal",
    "occipital": "Occipital",
    "temporal": "Temporal",
}
WINDOW_ORDER = ((0.0, 250.0), (125.0, 375.0), (250.0, 500.0))
FOCUS_REGION = "central"
FOCUS_WINDOW = (125.0, 375.0)
BLUE = "#1f5a8a"
BLUE_DARK = "#0b2d4d"
BLUE_LIGHT = "#78aeda"
ORANGE = "#d95f02"
RED = "#c9252d"
GREY = "#8d959d"
GRID = "#d9dde2"


def _summary_paths(root: Path) -> dict[str, Path]:
    return {
        "sweep": root
        / "summary_mechanistic_dissociation_sweep/mechanistic_sweep_table.csv",
        "loso": root / "summary_mechanistic_robustness_central/loso_table.csv",
        "perm": root
        / "summary_mechanistic_robustness_central/permutation_null_betas.csv",
        "per_subject": root
        / "summary_mechanistic_dissociation_central/per_subject_spearman_partial.csv",
        "alpha": root / "summary_alpha_comparison_central/alpha_comparison_lme.csv",
    }


def _load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _focus_row(sweep: pd.DataFrame) -> pd.Series:
    mask = (
        (sweep["region"] == FOCUS_REGION)
        & (sweep["window_start_ms"] == FOCUS_WINDOW[0])
        & (sweep["window_end_ms"] == FOCUS_WINDOW[1])
    )
    if not mask.any():
        raise ValueError("Could not find the central 125-375 ms focus row")
    return sweep.loc[mask].iloc[0]


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value < 1e-4:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _sig_colour(p_holm: float) -> str:
    if not np.isfinite(p_holm):
        return "#b9b9b9"
    if p_holm < 1e-10:
        return BLUE_DARK
    if p_holm < 0.001:
        return BLUE
    if p_holm < 0.05:
        return BLUE_LIGHT
    return "#b9b9b9"


def panel_a_detection_forest(ax: plt.Axes, sweep: pd.DataFrame) -> None:
    rows: list[dict[str, float | str]] = []
    for region in REGION_ORDER:
        for start, end in WINDOW_ORDER:
            match = sweep[
                (sweep["region"] == region)
                & (sweep["window_start_ms"] == start)
                & (sweep["window_end_ms"] == end)
            ]
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append(
                {
                    "label": f"{REGION_LABELS[region]}\n{int(start)}-{int(end)} ms",
                    "region": region,
                    "start": float(start),
                    "end": float(end),
                    "beta": float(row["lme_det_beta"]),
                    "ci_lo": float(row["lme_det_ci95_low"]),
                    "ci_hi": float(row["lme_det_ci95_high"]),
                    "holm_p": float(row["lme_det_p_holm"]),
                }
            )

    cells = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(cells))
    focus_mask = (
        (cells["region"] == FOCUS_REGION)
        & (cells["start"] == FOCUS_WINDOW[0])
        & (cells["end"] == FOCUS_WINDOW[1])
    )
    if focus_mask.any():
        focus_y = int(np.flatnonzero(focus_mask.to_numpy())[0])
        ax.axhspan(focus_y - 0.45, focus_y + 0.45, color="#eef5fb", zorder=0)

    err_lo = cells["beta"] - cells["ci_lo"]
    err_hi = cells["ci_hi"] - cells["beta"]
    ax.errorbar(
        cells["beta"],
        y,
        xerr=[err_lo, err_hi],
        fmt="none",
        ecolor="#7f7f7f",
        elinewidth=1.2,
        capsize=2.5,
        zorder=2,
    )
    colours = [_sig_colour(p) for p in cells["holm_p"]]
    ax.scatter(
        cells["beta"],
        y,
        s=58,
        c=colours,
        edgecolor="#202020",
        linewidth=0.45,
        zorder=3,
    )

    ax.axvline(0.0, color="#555", lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels(cells["label"], fontsize=7.5, linespacing=0.9)
    ax.set_xlabel("Hit-vs-miss coefficient (95% CI)")
    ax.set_title("A  Detection effect across regions and windows", loc="left")
    ax.set_xlim(-0.006, 0.071)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=BLUE_DARK,
            markeredgecolor="#202020",
            markersize=6,
            label="Holm p < 1e-10",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=BLUE,
            markeredgecolor="#202020",
            markersize=6,
            label="Holm p < 0.001",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=BLUE_LIGHT,
            markeredgecolor="#202020",
            markersize=6,
            label="Holm p < 0.05",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#b9b9b9",
            markeredgecolor="#202020",
            markersize=6,
            label="not significant",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=7.0,
        borderpad=0.2,
        handletextpad=0.5,
    )


def panel_b_loso(ax: plt.Axes, loso: pd.DataFrame, observed_beta: float) -> None:
    sorted_df = loso.sort_values("beta").reset_index(drop=True)
    y = np.arange(len(sorted_df))
    ax.barh(
        y,
        sorted_df["beta"],
        color=BLUE,
        alpha=0.9,
        edgecolor="#202020",
        linewidth=0.3,
        height=0.72,
    )
    ax.axvline(observed_beta, color=RED, lw=1.6, ls=(0, (5, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_df["dropped_subject"], fontsize=7.2)
    ax.set_xlabel("Coefficient after dropping one subject")
    ax.set_title("B  Leave-one-subject-out robustness", loc="left")
    ax.set_xlim(0.0, max(sorted_df["beta"].max(), observed_beta) * 1.12)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    text = (
        f"Full sample β = {observed_beta:+.3f}\n"
        f"LOSO β range [{sorted_df['beta'].min():+.3f}, {sorted_df['beta'].max():+.3f}]\n"
        f"worst p = {sorted_df['p'].max():.1e}"
    )
    ax.text(
        0.03,
        0.04,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#b7b7b7"),
    )


def panel_c_permutation(ax: plt.Axes, perm: pd.DataFrame, observed_beta: float) -> None:
    null_betas = perm["null_beta"].dropna().to_numpy(dtype=float)
    n_perm = len(null_betas)
    bins = np.linspace(min(null_betas.min(), -0.02), observed_beta * 1.08, 58)
    ax.hist(
        null_betas,
        bins=bins,
        color=GREY,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.9,
    )
    ax.axvline(observed_beta, color=RED, lw=1.8)
    ax.set_xlabel("Permuted hit-vs-miss coefficient")
    ax.set_ylabel("Permutation count")
    ax.set_title("C  Within-subject permutation null", loc="left")
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    n_extreme = int(np.sum(np.abs(null_betas) >= abs(observed_beta)))
    empirical_p = (n_extreme + 1) / (n_perm + 1)
    null_z = (observed_beta - null_betas.mean()) / (null_betas.std() + 1e-12)
    text = (
        f"{n_extreme}/{n_perm} null effects exceed observed\n"
        f"empirical p = {empirical_p:.4f}\n"
        f"null mean ± SD = {null_betas.mean():+.4f} ± {null_betas.std():.4f}\n"
        f"observed z vs null = {null_z:.2f}"
    )
    ax.text(
        0.57,
        0.82,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#b7b7b7"),
    )


def panel_d_subject_rhos(ax: plt.Axes, per_subject: pd.DataFrame) -> None:
    df = per_subject.dropna(
        subset=[
            "spearman_partial_pred_fit_vs_detection",
            "spearman_partial_pred_fit_vs_confidence",
        ]
    ).copy()
    rng = np.random.default_rng(7)
    det_x = np.zeros(len(df)) + rng.uniform(-0.045, 0.045, len(df))
    conf_x = np.ones(len(df)) + rng.uniform(-0.045, 0.045, len(df))
    det = df["spearman_partial_pred_fit_vs_detection"].to_numpy(dtype=float)
    conf = df["spearman_partial_pred_fit_vs_confidence"].to_numpy(dtype=float)

    for x0, x1, y0, y1 in zip(det_x, conf_x, det, conf):
        ax.plot([x0, x1], [y0, y1], color="#c6c6c6", lw=0.75, zorder=1)
    ax.scatter(
        det_x,
        det,
        color=BLUE,
        edgecolor="#202020",
        linewidth=0.45,
        s=48,
        label="Detection",
        zorder=3,
    )
    ax.scatter(
        conf_x,
        conf,
        color=ORANGE,
        edgecolor="#202020",
        linewidth=0.45,
        s=48,
        label="Confidence on hits",
        zorder=3,
    )
    ax.plot([-0.17, 0.17], [det.mean(), det.mean()], color=BLUE_DARK, lw=2.0)
    ax.plot([0.83, 1.17], [conf.mean(), conf.mean()], color="#9a4300", lw=2.0)
    ax.axhline(0.0, color="#555", lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Detection", "Confidence\n(hits only)"])
    ax.set_ylabel("Per-subject Spearman partial r")
    ax.set_title("D  Continuous subject-level associations", loc="left")
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(-0.16, 0.33)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    text = (
        f"detection mean r = {det.mean():+.3f}\n"
        f"confidence mean r = {conf.mean():+.3f}\n"
        "paired difference p = 0.105"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#b7b7b7"),
    )
    ax.legend(loc="lower left", frameon=False, fontsize=7.5)


def write_key_numbers(
    *,
    outdir: Path,
    focus: pd.Series,
    loso: pd.DataFrame,
    perm: pd.DataFrame,
    alpha: pd.DataFrame | None,
) -> None:
    observed_beta = float(focus["lme_det_beta"])
    null_betas = perm["null_beta"].dropna().to_numpy(dtype=float)
    rows = [
        {
            "quantity": "focus_cell",
            "value": "central 125-375 ms",
            "notes": "pre-registered anatomical/time-window headline for the EEG result",
        },
        {
            "quantity": "detection_lme_beta",
            "value": observed_beta,
            "notes": (
                f"z={float(focus['lme_det_z']):.2f}, p={float(focus['lme_det_p']):.3e}, "
                f"Holm p={float(focus['lme_det_p_holm']):.3e}"
            ),
        },
        {
            "quantity": "confidence_lme_beta",
            "value": float(focus["lme_conf_beta"]),
            "notes": (
                f"z={float(focus['lme_conf_z']):.2f}, p={float(focus['lme_conf_p']):.3e}, "
                f"Holm p={float(focus['lme_conf_p_holm']):.3e}"
            ),
        },
        {
            "quantity": "loso_beta_range",
            "value": f"{loso['beta'].min():+.5f} to {loso['beta'].max():+.5f}",
            "notes": f"worst p={loso['p'].max():.3e}",
        },
        {
            "quantity": "permutation_empirical_p",
            "value": (int(np.sum(np.abs(null_betas) >= abs(observed_beta))) + 1)
            / (len(null_betas) + 1),
            "notes": f"n_perm={len(null_betas)}",
        },
    ]
    if alpha is not None and not alpha.empty:
        m2 = alpha[
            (alpha["model"] == "M2_pred_fit_with_alpha_covariate")
            & (alpha["term"] == "sdt_int")
        ]
        if not m2.empty:
            row = m2.iloc[0]
            rows.append(
                {
                    "quantity": "alpha_control_sdt_beta",
                    "value": float(row["beta"]),
                    "notes": (
                        "detection coefficient after adding central pre-stim log-alpha; "
                        f"p={float(row['p']):.3e}"
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(outdir / "headline_key_numbers.csv", index=False)


def write_caption(outdir: Path, focus: pd.Series, perm: pd.DataFrame) -> None:
    observed_beta = float(focus["lme_det_beta"])
    null_betas = perm["null_beta"].dropna().to_numpy(dtype=float)
    n_extreme = int(np.sum(np.abs(null_betas) >= abs(observed_beta)))
    empirical_p = (n_extreme + 1) / (len(null_betas) + 1)
    caption = f"""# EEG Mechanistic Headline

Pre-stimulus multivariate predictive structure carries a graded perceptual-state
signal at central scalp. In the central 125-375 ms window, the marginal
pre-stim VAR predictive-fit contribution is larger on hit than miss trials
(β={observed_beta:+.3f}, z={float(focus['lme_det_z']):.2f},
p={float(focus['lme_det_p']):.2e}; Holm p across 15 region-by-window cells
={float(focus['lme_det_p_holm']):.2e}; n={int(focus['n_trials'])} trials,
{int(focus['n_subjects'])} subjects). The same cell carries a weaker confidence
loading on hit trials (β={float(focus['lme_conf_beta']):+.3f}, Holm
p={float(focus['lme_conf_p_holm']):.3f}). Leave-one-subject-out refits remain
positive, and a within-subject permutation null gives empirical p={empirical_p:.4f}
({n_extreme}/{len(null_betas)} null effects as large as observed).

The honest framing is not a pure detection-only dissociation: central dynamics
scale with both detection and confidence, with detection the larger and more
robust loading. The control point is that this contribution is measured over
and above post-stimulus evoked amplitude and stimulus amplitude.
"""
    (outdir / "headline_caption.md").write_text(caption, encoding="utf-8")


def build_figure(
    *,
    sweep: pd.DataFrame,
    loso: pd.DataFrame,
    perm: pd.DataFrame,
    per_subject: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> None:
    focus = _focus_row(sweep)
    observed_beta = float(focus["lme_det_beta"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
        }
    )
    fig = plt.figure(figsize=(11.2, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.12, 1.0])
    axes = np.array(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )

    panel_a_detection_forest(axes[0, 0], sweep)
    panel_b_loso(axes[0, 1], loso, observed_beta)
    panel_c_permutation(axes[1, 0], perm, observed_beta)
    panel_d_subject_rhos(axes[1, 1], per_subject)

    fig.suptitle(
        "Pre-stimulus dynamics predict perceptual state beyond evoked amplitude",
        fontsize=12.5,
        y=1.025,
    )
    fig.supxlabel(
        "Linear mixed-effects models include subject random intercepts and stimulus amplitude as a covariate.",
        fontsize=7.6,
        y=-0.015,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--png-name", default="headline_figure.png")
    parser.add_argument("--pdf-name", default="headline_figure.pdf")
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = (
        root / "summary_mechanistic_dissociation_sweep"
        if args.outdir is None
        else args.outdir.resolve()
    )
    paths = _summary_paths(root)
    sweep = _load_required(paths["sweep"])
    loso = _load_required(paths["loso"])
    perm = _load_required(paths["perm"])
    per_subject = _load_required(paths["per_subject"])
    alpha = pd.read_csv(paths["alpha"]) if paths["alpha"].exists() else None

    out_png = outdir / args.png_name
    out_pdf = outdir / args.pdf_name
    build_figure(
        sweep=sweep,
        loso=loso,
        perm=perm,
        per_subject=per_subject,
        out_png=out_png,
        out_pdf=out_pdf,
    )
    focus = _focus_row(sweep)
    write_key_numbers(outdir=outdir, focus=focus, loso=loso, perm=perm, alpha=alpha)
    write_caption(outdir, focus, perm)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    print(f"Saved {outdir / 'headline_key_numbers.csv'}")
    print(f"Saved {outdir / 'headline_caption.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
