"""Summarise subject-level PRISM EEG region-window results."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _eeg_stats import _ci95, _format_p, _holm
from eeg_prism_timecourse_decoder import (
    _window_label,
    discover_window_dirs,
    run_window,
)
from eeg_subject_decoder import (
    DEFAULT_BASELINE_END_MS,
    DEFAULT_BASELINE_RESULTS,
    DEFAULT_BASELINE_START_MS,
    DEFAULT_DERIVATIVES_DIR,
    DEFAULT_EXPORT_DIR,
    DEFAULT_N_FOLDS,
    DEFAULT_PRISM_MODEL_FAMILY,
    DEFAULT_REP_DIM,
)


DEFAULT_ROOT = Path("data/results_prism/eeg_prism_region_timecourse_pca_q4")
DEFAULT_OUTDIR = DEFAULT_ROOT / "summary_subject_decoder_region_timecourse"
DEFAULT_REGIONS = ("central", "frontal", "parietal", "occipital", "temporal")
CONTRIBUTION_MODELS = {
    "prism_contribution": ("raw_central_delta", "raw_plus_prism"),
    "var_contribution": ("raw_central_delta", "raw_plus_var"),
}


def _add_holm_columns(df: pd.DataFrame, *, group_cols: list[str], p_col: str = "ttest_p") -> pd.DataFrame:
    out = df.copy()
    out[f"{p_col}_holm"] = np.nan
    for _, group in out.groupby(group_cols, dropna=False, sort=False):
        out.loc[group.index, f"{p_col}_holm"] = _holm(group[p_col]).to_numpy(dtype=float)
    return out


def _add_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    out = df.copy()
    out["region_name"] = region
    return out


def _subject_contribution_summary(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_cols = [
        "region_name",
        "window_label",
        "target_center_ms",
        "subject",
        "trial_idx",
        "label",
        "confidence",
    ]
    wide = scores.pivot_table(index=index_cols, columns="model_name", values="decoder_score").reset_index()
    subject_rows: list[dict[str, object]] = []

    for contrast, (raw_model, hybrid_model) in CONTRIBUTION_MODELS.items():
        if raw_model not in wide.columns or hybrid_model not in wide.columns:
            continue
        wide[contrast] = wide[hybrid_model] - wide[raw_model]
        for (region, window, centre, subject), group in wide.groupby(
            ["region_name", "window_label", "target_center_ms", "subject"],
            sort=True,
        ):
            hit = group.loc[group["label"].eq(1), contrast].dropna().to_numpy(dtype=float)
            miss = group.loc[group["label"].eq(0), contrast].dropna().to_numpy(dtype=float)
            if hit.size == 0 or miss.size == 0:
                continue
            hit_rows = group.loc[group["label"].eq(1), [contrast, "confidence"]].dropna()
            rho = math.nan
            if hit_rows.shape[0] >= 5:
                rho = float(stats.spearmanr(hit_rows[contrast], hit_rows["confidence"]).statistic)
            subject_rows.append(
                {
                    "contrast": contrast,
                    "region_name": region,
                    "window_label": window,
                    "target_center_ms": centre,
                    "subject": subject,
                    "hit_minus_miss_contribution": float(hit.mean() - miss.mean()),
                    "confidence_rho_hits": rho,
                }
            )

    subject_df = pd.DataFrame(subject_rows)
    group_rows: list[dict[str, object]] = []
    for keys, group in subject_df.groupby(["contrast", "region_name", "window_label", "target_center_ms"], sort=True):
        contrast, region, window, centre = keys
        for metric in ("hit_minus_miss_contribution", "confidence_rho_hits"):
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            ci_low, ci_high = _ci95(values)
            t = stats.ttest_1samp(values, 0.0) if values.size >= 2 else None
            try:
                wilcoxon_p = float(stats.wilcoxon(values).pvalue)
            except ValueError:
                wilcoxon_p = math.nan
            group_rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "region_name": region,
                    "window_label": window,
                    "target_center_ms": centre,
                    "n_subjects": int(values.size),
                    "mean": float(values.mean()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "positive_subjects": int(np.sum(values > 0.0)),
                    "ttest_p": float(t.pvalue) if t is not None else math.nan,
                    "wilcoxon_p": wilcoxon_p,
                }
            )
    return subject_df, pd.DataFrame(group_rows)


def _plot_heatmaps(contrib_summary: pd.DataFrame, outdir: Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism-mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "prism-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    focus = contrib_summary.loc[
        contrib_summary["contrast"].eq("prism_contribution")
        & contrib_summary["metric"].eq("hit_minus_miss_contribution")
    ].copy()
    if focus.empty:
        return

    regions = list(DEFAULT_REGIONS)
    windows = sorted(focus["window_label"].unique(), key=lambda value: focus.loc[focus["window_label"].eq(value), "target_center_ms"].iloc[0])
    mean_grid = focus.pivot(index="region_name", columns="window_label", values="mean").reindex(index=regions, columns=windows)
    p_grid = focus.pivot(index="region_name", columns="window_label", values="ttest_p_holm").reindex(index=regions, columns=windows)

    vmax = float(np.nanmax(np.abs(mean_grid.to_numpy(dtype=float))))
    vmax = max(vmax, 1e-6)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    image = ax.imshow(mean_grid.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(windows)), windows, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(regions)), regions)
    ax.set_xlabel("Post-stimulus window (ms)")
    ax.set_title("PRISM contribution to hit-vs-miss evidence")
    for row_idx, region in enumerate(regions):
        for col_idx, window in enumerate(windows):
            value = mean_grid.loc[region, window]
            p_value = p_grid.loc[region, window]
            if pd.isna(value):
                continue
            mark = "*" if p_value < 0.05 else ""
            ax.text(col_idx, row_idx, f"{value:.3f}{mark}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Mean contribution")
    fig.savefig(outdir / "eeg_prism_region_timecourse_heatmap.png", dpi=300)
    fig.savefig(outdir / "eeg_prism_region_timecourse_heatmap.pdf")
    plt.close(fig)


def _plot_summary_figure(
    group_summary: pd.DataFrame,
    contrib_summary: pd.DataFrame,
    outdir: Path,
    *,
    focus_region: str = "occipital",
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism-mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "prism-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    focus = contrib_summary.loc[
        contrib_summary["contrast"].eq("prism_contribution")
        & contrib_summary["metric"].eq("hit_minus_miss_contribution")
    ].copy()
    if focus.empty:
        return

    regions = list(DEFAULT_REGIONS)
    windows = sorted(
        focus["window_label"].unique(),
        key=lambda value: focus.loc[focus["window_label"].eq(value), "target_center_ms"].iloc[0],
    )
    mean_grid = focus.pivot(index="region_name", columns="window_label", values="mean").reindex(
        index=regions,
        columns=windows,
    )
    p_grid = focus.pivot(index="region_name", columns="window_label", values="ttest_p_holm").reindex(
        index=regions,
        columns=windows,
    )

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.35, 3.05),
        gridspec_kw={"width_ratios": [1.18, 1.32]},
        constrained_layout=True,
    )

    ax = axes[0]
    vmax = max(float(np.nanmax(np.abs(mean_grid.to_numpy(dtype=float)))), 1e-6)
    image = ax.imshow(mean_grid.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(windows)), windows, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(regions)), regions)
    ax.set_xlabel("Window (ms)")
    ax.set_title("PRISM contribution", fontsize=9)
    for row_idx, region in enumerate(regions):
        for col_idx, window in enumerate(windows):
            value = mean_grid.loc[region, window]
            p_value = p_grid.loc[region, window]
            if pd.isna(value):
                continue
            mark = "*" if p_value < 0.05 else ""
            colour = "white" if abs(float(value)) > 0.65 * vmax else "#111111"
            ax.text(col_idx, row_idx, f"{value:.3f}{mark}", ha="center", va="center", fontsize=7, color=colour)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    ax.text(-0.18, 1.05, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = axes[1]
    model_labels = {
        "raw_central_delta": "Raw EEG",
        "raw_plus_var": "Raw + VAR",
        "raw_plus_prism": "Raw + PRISM",
    }
    colours = {
        "raw_central_delta": "#56616c",
        "raw_plus_var": "#d56a1c",
        "raw_plus_prism": "#2f7f7f",
    }
    region_df = group_summary.loc[
        group_summary["region_name"].eq(focus_region)
        & group_summary["model_name"].isin(model_labels)
    ].copy()
    for model, label in model_labels.items():
        model_df = region_df.loc[region_df["model_name"].eq(model)].sort_values("target_center_ms")
        x = model_df["target_center_ms"].to_numpy(dtype=float)
        y = model_df["auc_mean"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", markersize=4.0, linewidth=1.7, color=colours[model], label=label)
    ax.axhline(0.5, color="#707780", linewidth=0.9, linestyle="--")
    ticks = (
        region_df[["target_center_ms", "window_label"]]
        .drop_duplicates()
        .sort_values("target_center_ms")
    )
    ax.set_xticks(
        ticks["target_center_ms"].to_numpy(dtype=float),
        ticks["window_label"].tolist(),
        rotation=30,
        ha="right",
    )
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel("Held-out AUC")
    ax.set_title(f"{focus_region.capitalize()} decoder", fontsize=9)
    ax.grid(axis="y", color="#e8eaed", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.savefig(outdir / "eeg_prism_region_timecourse_summary.png", dpi=300)
    fig.savefig(outdir / "eeg_prism_region_timecourse_summary.pdf")
    plt.close(fig)


def _write_ranked_cell_report(
    group_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    contrib_summary: pd.DataFrame,
    outpath: Path,
) -> None:
    n_cells = int(
        contrib_summary.loc[
            contrib_summary["contrast"].eq("prism_contribution")
            & contrib_summary["metric"].eq("hit_minus_miss_contribution"),
            ["region_name", "window_label"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    prism_contrib = contrib_summary.loc[
        contrib_summary["contrast"].eq("prism_contribution")
        & contrib_summary["metric"].eq("hit_minus_miss_contribution")
    ].sort_values(["ttest_p_holm", "ttest_p", "mean"], ascending=[True, True, False])
    prism_conf = contrib_summary.loc[
        contrib_summary["contrast"].eq("prism_contribution")
        & contrib_summary["metric"].eq("confidence_rho_hits")
    ].sort_values(["ttest_p_holm", "ttest_p", "mean"], ascending=[True, True, False])
    prism_auc = pairwise.loc[pairwise["contrast"].eq("prism_gain_over_raw")].sort_values(
        ["ttest_p_holm", "ttest_p", "mean_delta_model_b_minus_a"], ascending=[True, True, False]
    )
    var_minus_prism = pairwise.loc[pairwise["contrast"].eq("var_minus_prism_hybrid")].sort_values(
        ["ttest_p_holm", "ttest_p", "mean_delta_model_b_minus_a"], ascending=[True, True, False]
    )
    prism_standalone = group_summary.loc[group_summary["model_name"].eq("prism_central_augmented")].sort_values(
        ["auc_ttest_vs_half_p_holm", "auc_ttest_vs_half_p", "auc_mean"], ascending=[True, True, False]
    )

    lines = [
        "# PRISM EEG Region-Window Ranking",
        "",
        f"Cells are ranked after treating subject as the unit of inference. Holm correction is applied within each family of {n_cells} region-window tests.",
        "",
        "## PRISM contribution beyond raw EEG",
        "",
    ]
    for _, row in prism_contrib.head(8).iterrows():
        lines.append(
            f"- {row['region_name']} {row['window_label']} ms: mean hit-minus-miss contribution="
            f"{row['mean']:.4f}, 95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], "
            f"{int(row['positive_subjects'])}/{int(row['n_subjects'])} positive subjects, "
            f"paired t p={_format_p(float(row['ttest_p']))}, Holm p={_format_p(float(row['ttest_p_holm']))}, "
            f"Wilcoxon p={_format_p(float(row['wilcoxon_p']))}."
        )
    lines.extend(["", "## Raw + PRISM AUC gain over raw", ""])
    for _, row in prism_auc.head(8).iterrows():
        lines.append(
            f"- {row['region_name']} {row['window_label']} ms: delta AUC="
            f"{row['mean_delta_model_b_minus_a']:.4f}, 95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], "
            f"{int(row['positive_subjects'])}/{int(row['n_subjects'])} positive subjects, "
            f"paired t p={_format_p(float(row['ttest_p']))}, Holm p={_format_p(float(row['ttest_p_holm']))}."
        )
    lines.extend(["", "## Standalone PRISM decoder", ""])
    for _, row in prism_standalone.head(8).iterrows():
        lines.append(
            f"- {row['region_name']} {row['window_label']} ms: mean AUC={row['auc_mean']:.4f}, "
            f"95% CI [{row['auc_ci95_low']:.4f}, {row['auc_ci95_high']:.4f}], "
            f"p={_format_p(float(row['auc_ttest_vs_half_p']))}, Holm p={_format_p(float(row['auc_ttest_vs_half_p_holm']))}."
        )
    lines.extend(["", "## Confidence on hits", ""])
    for _, row in prism_conf.head(8).iterrows():
        lines.append(
            f"- {row['region_name']} {row['window_label']} ms: mean Spearman rho={row['mean']:.4f}, "
            f"95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], "
            f"{int(row['positive_subjects'])}/{int(row['n_subjects'])} positive subjects, "
            f"paired t p={_format_p(float(row['ttest_p']))}, Holm p={_format_p(float(row['ttest_p_holm']))}."
        )
    lines.extend(["", "## VAR minus PRISM hybrid", ""])
    for _, row in var_minus_prism.head(8).iterrows():
        lines.append(
            f"- {row['region_name']} {row['window_label']} ms: delta AUC={row['mean_delta_model_b_minus_a']:.4f}, "
            f"95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], "
            f"{int(row['positive_subjects'])}/{int(row['n_subjects'])} positive subjects, "
            f"paired t p={_format_p(float(row['ttest_p']))}, Holm p={_format_p(float(row['ttest_p_holm']))}."
        )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    root: Path,
    baseline_results_dir: Path,
    derivatives_dir: Path,
    export_dir: Path,
    outdir: Path,
    regions: tuple[str, ...],
    rep_dim: int,
    baseline_start_ms: float,
    baseline_end_ms: float,
    prism_model_family: str,
    positive_sdt: str,
    negative_sdt: str,
    n_folds: int,
    ridge: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    score_parts: list[pd.DataFrame] = []
    subject_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    pairwise_parts: list[pd.DataFrame] = []

    for window_dir, start_ms, end_ms in discover_window_dirs(root):
        for region in regions:
            scores, subject_summary, group_summary, pairwise = run_window(
                window_dir=window_dir,
                start_ms=start_ms,
                end_ms=end_ms,
                baseline_results_dir=baseline_results_dir,
                derivatives_dir=derivatives_dir,
                export_dir=export_dir,
                region=region,
                rep_dim=rep_dim,
                baseline_start_ms=baseline_start_ms,
                baseline_end_ms=baseline_end_ms,
                prism_model_family=prism_model_family,
                positive_sdt=positive_sdt,
                negative_sdt=negative_sdt,
                n_folds=n_folds,
                ridge=ridge,
            )
            score_parts.append(_add_region(scores, region))
            subject_parts.append(_add_region(subject_summary, region))
            group_parts.append(_add_region(group_summary, region))
            pairwise_parts.append(_add_region(pairwise, region))

    scores = pd.concat(score_parts, ignore_index=True)
    subject_summary = pd.concat(subject_parts, ignore_index=True)
    group_summary = pd.concat(group_parts, ignore_index=True)
    pairwise = pd.concat(pairwise_parts, ignore_index=True)
    contribution_subject, contribution_group = _subject_contribution_summary(scores)
    group_summary = _add_holm_columns(
        group_summary,
        group_cols=["model_name"],
        p_col="auc_ttest_vs_half_p",
    )
    pairwise = _add_holm_columns(pairwise, group_cols=["contrast"], p_col="ttest_p")
    contribution_group = _add_holm_columns(
        contribution_group,
        group_cols=["contrast", "metric"],
        p_col="ttest_p",
    )

    scores.to_csv(outdir / "eeg_prism_region_timecourse_trial_scores.csv", index=False)
    subject_summary.to_csv(outdir / "eeg_prism_region_timecourse_subject_summary.csv", index=False)
    group_summary.to_csv(outdir / "eeg_prism_region_timecourse_group_summary.csv", index=False)
    pairwise.to_csv(outdir / "eeg_prism_region_timecourse_pairwise.csv", index=False)
    contribution_subject.to_csv(outdir / "eeg_prism_region_timecourse_contribution_subject.csv", index=False)
    contribution_group.to_csv(outdir / "eeg_prism_region_timecourse_contribution_summary.csv", index=False)
    _plot_heatmaps(contribution_group, outdir)
    _plot_summary_figure(group_summary, contribution_group, outdir)
    _write_ranked_cell_report(
        group_summary,
        pairwise,
        contribution_group,
        outdir / "eeg_prism_region_timecourse_ranked_cells.md",
    )
    print(f"Wrote {outdir / 'eeg_prism_region_timecourse_ranked_cells.md'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline-results-dir", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--derivatives-dir", type=Path, default=DEFAULT_DERIVATIVES_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    parser.add_argument("--rep-dim", type=int, default=DEFAULT_REP_DIM)
    parser.add_argument("--baseline-start-ms", type=float, default=DEFAULT_BASELINE_START_MS)
    parser.add_argument("--baseline-end-ms", type=float, default=DEFAULT_BASELINE_END_MS)
    parser.add_argument("--prism-model-family", default=DEFAULT_PRISM_MODEL_FAMILY)
    parser.add_argument("--positive-sdt", default="hit")
    parser.add_argument("--negative-sdt", default="miss")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--ridge", type=float, default=1e-3)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run(
        root=args.root,
        baseline_results_dir=args.baseline_results_dir,
        derivatives_dir=args.derivatives_dir,
        export_dir=args.export_dir,
        outdir=args.outdir,
        regions=tuple(args.regions),
        rep_dim=args.rep_dim,
        baseline_start_ms=args.baseline_start_ms,
        baseline_end_ms=args.baseline_end_ms,
        prism_model_family=args.prism_model_family,
        positive_sdt=args.positive_sdt,
        negative_sdt=args.negative_sdt,
        n_folds=args.n_folds,
        ridge=args.ridge,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
