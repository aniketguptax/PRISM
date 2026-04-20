"""Relate predictive-fit topography to matched-evidence topography."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from framework.summaries import prepare_pyplot, save_table


DEFAULT_REGION_CONTROL_SUMMARY_CSV = Path(
    "./data/results_baseline/region_sliding_baseline300ms_controls_focus_q4/summary/"
    "region_sliding_control_group_summary_q4.csv"
)
DEFAULT_MATCHED_SPATIAL_SWEEP_CSV = Path(
    "./data/results_baseline/region_sliding_baseline300ms_controls_focus_q4/"
    "summary_matched_spatial_control_sweep_full/matched_spatial_control_sweep_hybrid_summary.csv"
)
POSTERIOR_REGIONS = {"parietal", "occipital"}


def load_inputs(
    region_control_summary_csv: Path,
    matched_spatial_sweep_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the predictive-fit and matched-evidence summaries."""
    if not region_control_summary_csv.exists():
        raise FileNotFoundError(f"Region-control summary not found: {region_control_summary_csv}")
    if not matched_spatial_sweep_csv.exists():
        raise FileNotFoundError(f"Matched spatial sweep summary not found: {matched_spatial_sweep_csv}")

    control_df = pd.read_csv(region_control_summary_csv)
    sweep_df = pd.read_csv(matched_spatial_sweep_csv)
    required_control = {"matched_region_name", "target_start_ms", "target_end_ms", "mean_delta_pred_r2_obs"}
    required_sweep = {"focus_region", "target_start_ms", "target_end_ms", "pair_acc_minus_control_mean", "mean_margin_minus_control_mean"}
    missing_control = sorted(required_control.difference(control_df.columns))
    missing_sweep = sorted(required_sweep.difference(sweep_df.columns))
    if missing_control:
        raise ValueError(f"Region-control summary is missing columns: {missing_control}")
    if missing_sweep:
        raise ValueError(f"Matched spatial sweep summary is missing columns: {missing_sweep}")
    return control_df, sweep_df


def build_dissociation_table(
    control_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join predictive-fit and evidence summaries on region and window."""
    merged = control_df.merge(
        sweep_df,
        left_on=["matched_region_name", "target_start_ms", "target_end_ms"],
        right_on=["focus_region", "target_start_ms", "target_end_ms"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping region/window rows were found")

    merged["is_posterior"] = merged["focus_region"].isin(POSTERIOR_REGIONS)
    merged["window_label"] = merged["window_label"].astype(str)
    merged["point_label"] = (
        merged["focus_region"].astype(str)
        + " "
        + merged["window_label"].astype(str)
    )
    return merged.sort_values(["focus_region", "target_start_ms"]).reset_index(drop=True)


def summarise_metric_relation(
    merged_df: pd.DataFrame,
    *,
    evidence_col: str,
) -> dict[str, float]:
    """Quantify one predictive-fit versus evidence relation."""
    x = merged_df["mean_delta_pred_r2_obs"].to_numpy(dtype=float)
    y = merged_df[evidence_col].to_numpy(dtype=float)
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)

    posterior = merged_df.loc[merged_df["is_posterior"], evidence_col].to_numpy(dtype=float)
    other = merged_df.loc[~merged_df["is_posterior"], evidence_col].to_numpy(dtype=float)
    posterior_t = stats.ttest_ind(posterior, other, equal_var=False)

    return {
        "metric": evidence_col,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "posterior_mean": float(np.mean(posterior)),
        "other_mean": float(np.mean(other)),
        "posterior_minus_other": float(np.mean(posterior) - np.mean(other)),
        "posterior_vs_other_ttest_p": float(posterior_t.pvalue),
        "n_points": int(len(x)),
        "n_posterior": int(len(posterior)),
        "n_other": int(len(other)),
    }


def build_relation_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Create the summary table for both evidence metrics."""
    rows = [
        summarise_metric_relation(merged_df, evidence_col="pair_acc_minus_control_mean"),
        summarise_metric_relation(merged_df, evidence_col="mean_margin_minus_control_mean"),
    ]
    return pd.DataFrame(rows)


def plot_dissociation(
    merged_df: pd.DataFrame,
    *,
    outfile: Path,
) -> None:
    """Plot predictive-fit advantage against evidence advantage."""
    if merged_df.empty:
        return

    colours = {
        "frontal": "#8d99ae",
        "central": "#d62828",
        "temporal": "#457b9d",
        "parietal": "#2a9d8f",
        "occipital": "#6a4c93",
    }
    markers = {
        "0-250": "o",
        "125-375": "s",
        "250-500": "^",
    }
    plot_specs = [
        ("pair_acc_minus_control_mean", "Hybrid pair-accuracy minus control"),
        ("mean_margin_minus_control_mean", "Hybrid score margin minus control"),
    ]

    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for axis, (metric_col, title) in zip(axes, plot_specs):
        for row in merged_df.itertuples(index=False):
            colour = colours.get(str(row.focus_region), "#495057")
            marker = markers.get(str(row.window_label), "o")
            axis.scatter(
                float(row.mean_delta_pred_r2_obs),
                float(getattr(row, metric_col)),
                color=colour,
                marker=marker,
                s=70,
                alpha=0.9,
            )
            axis.annotate(
                str(row.point_label),
                (float(row.mean_delta_pred_r2_obs), float(getattr(row, metric_col))),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )

        x = merged_df["mean_delta_pred_r2_obs"].to_numpy(dtype=float)
        y = merged_df[metric_col].to_numpy(dtype=float)
        coeffs = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        y_line = coeffs[0] * x_line + coeffs[1]
        axis.plot(x_line, y_line, color="#212529", linewidth=1.5, linestyle="--")
        axis.axhline(0.0, color="#adb5bd", linewidth=1.0)
        axis.axvline(0.0, color="#adb5bd", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Predictive-fit advantage over controls (mean delta R^2)")
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_dissociation_report(summary_df: pd.DataFrame) -> str:
    """Write a compact interpretation of the dissociation statistics."""
    lines = [
        "# Spatial Dissociation Summary",
        "",
    ]
    for row in summary_df.itertuples(index=False):
        metric_label = (
            "pair-accuracy advantage"
            if row.metric == "pair_acc_minus_control_mean"
            else "score-margin advantage"
        )
        lines.extend(
            [
                f"## {metric_label}",
                "",
                (
                    f"- Pearson correlation with predictive-fit advantage: "
                    f"r={row.pearson_r:.3f}, p={row.pearson_p:.3g}."
                ),
                (
                    f"- Spearman correlation with predictive-fit advantage: "
                    f"rho={row.spearman_rho:.3f}, p={row.spearman_p:.3g}."
                ),
                (
                    f"- Posterior regions minus the other regions: "
                    f"{row.posterior_minus_other:.3f} "
                    f"(posterior mean={row.posterior_mean:.3f}, other mean={row.other_mean:.3f}, "
                    f"Welch t p={row.posterior_vs_other_ttest_p:.3g})."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def run_spatial_dissociation_summary(
    *,
    region_control_summary_csv: Path = DEFAULT_REGION_CONTROL_SUMMARY_CSV,
    matched_spatial_sweep_csv: Path = DEFAULT_MATCHED_SPATIAL_SWEEP_CSV,
    outdir: Path | None = None,
) -> int:
    """Run the predictive-fit versus evidence dissociation summary."""
    outdir = (
        matched_spatial_sweep_csv.parent / "summary_spatial_dissociation"
        if outdir is None
        else outdir
    )
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        control_df, sweep_df = load_inputs(
            region_control_summary_csv,
            matched_spatial_sweep_csv,
        )
        merged_df = build_dissociation_table(control_df, sweep_df)
        summary_df = build_relation_summary(merged_df)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    merged_path = outdir / "spatial_dissociation_merged.csv"
    summary_path = outdir / "spatial_dissociation_summary.csv"
    plot_path = outdir / "spatial_dissociation_scatter.png"
    report_path = outdir / "spatial_dissociation_summary.md"

    save_table(merged_df, merged_path)
    save_table(summary_df, summary_path)
    plot_dissociation(merged_df, outfile=plot_path)
    report_path.write_text(build_dissociation_report(summary_df) + "\n", encoding="utf-8")

    print(f"Saved {merged_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {report_path}")
    return 0
