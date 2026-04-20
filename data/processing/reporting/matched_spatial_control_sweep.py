"""Batch matched spatial-control sweep across regions and target windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eegprep import DEFAULT_DERIVATIVES_DIR
from framework.summaries import prepare_pyplot, save_table
from reporting.evidence import DEFAULT_EXPORT_DIR, DEFAULT_NEGATIVE_SDT, DEFAULT_POSITIVE_SDT
from reporting.matched_spatial_control import (
    DEFAULT_BASELINE_RESULTS_DIR,
    DEFAULT_HYBRID_MODEL,
    DEFAULT_REP_DIM,
    run_matched_spatial_control_summary,
)
from reporting.statistics import DEFAULT_N_BOOT, DEFAULT_RANDOM_SEED, holm_adjust


DEFAULT_FOCUS_REGIONS = ("frontal", "central", "temporal", "parietal", "occipital")
DEFAULT_TARGET_WINDOWS_MS = (
    (0.0, 250.0),
    (125.0, 375.0),
    (250.0, 500.0),
)


def format_window_slug(start_ms: float, end_ms: float) -> str:
    """Build a stable directory slug for one target window."""
    return f"{int(start_ms)}_{int(end_ms)}"


def format_window_label(start_ms: float, end_ms: float) -> str:
    """Create a compact label for plots and tables."""
    return f"{int(start_ms)}-{int(end_ms)}"


def collect_sweep_outputs(
    *,
    baseline_results_dir: Path,
    derivatives_dir: Path,
    export_dir: Path,
    outdir: Path,
    rep_dim: int,
    focus_regions: tuple[str, ...],
    target_windows_ms: tuple[tuple[float, float], ...],
    positive_sdt: str,
    negative_sdt: str,
    random_seed: int,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one matched spatial-control summary per region/window pair and collect the tables."""
    model_frames: list[pd.DataFrame] = []
    gain_frames: list[pd.DataFrame] = []

    for focus_region in focus_regions:
        for target_start_ms, target_end_ms in target_windows_ms:
            combo_outdir = outdir / f"{focus_region}_{format_window_slug(target_start_ms, target_end_ms)}"
            status = run_matched_spatial_control_summary(
                baseline_results_dir=baseline_results_dir,
                derivatives_dir=derivatives_dir,
                export_dir=export_dir,
                outdir=combo_outdir,
                rep_dim=rep_dim,
                focus_region=focus_region,
                target_start_ms=target_start_ms,
                target_end_ms=target_end_ms,
                positive_sdt=positive_sdt,
                negative_sdt=negative_sdt,
                random_state=random_seed,
                n_bootstrap=n_bootstrap,
            )
            if status != 0:
                raise RuntimeError(
                    f"Matched spatial-control summary failed for {focus_region} "
                    f"{int(target_start_ms)}-{int(target_end_ms)} ms"
                )

            model_path = combo_outdir / "matched_spatial_control_model_vs_control_group.csv"
            gain_path = combo_outdir / "matched_spatial_control_named_vs_control_group.csv"
            model_df = pd.read_csv(model_path)
            gain_df = pd.read_csv(gain_path)
            model_df["focus_region"] = focus_region
            gain_df["focus_region"] = focus_region
            model_df["window_label"] = format_window_label(target_start_ms, target_end_ms)
            gain_df["window_label"] = format_window_label(target_start_ms, target_end_ms)
            model_frames.append(model_df)
            gain_frames.append(gain_df)

    return (
        pd.concat(model_frames, ignore_index=True),
        pd.concat(gain_frames, ignore_index=True),
    )


def apply_hybrid_family_corrections(model_summary: pd.DataFrame) -> pd.DataFrame:
    """Add Holm-corrected p-values across the region x window hybrid family."""
    out = model_summary.copy()
    hybrid_mask = out["model_name"].eq(DEFAULT_HYBRID_MODEL)
    out["pair_acc_minus_control_ttest_p_holm_hybrid"] = np.nan
    out["pair_acc_minus_control_wilcoxon_p_holm_hybrid"] = np.nan
    out["mean_margin_minus_control_ttest_p_holm_hybrid"] = np.nan
    out["mean_margin_minus_control_wilcoxon_p_holm_hybrid"] = np.nan

    out.loc[hybrid_mask, "pair_acc_minus_control_ttest_p_holm_hybrid"] = holm_adjust(
        out.loc[hybrid_mask, "pair_acc_minus_control_ttest_p"].to_list()
    )
    out.loc[hybrid_mask, "pair_acc_minus_control_wilcoxon_p_holm_hybrid"] = holm_adjust(
        out.loc[hybrid_mask, "pair_acc_minus_control_wilcoxon_p"].to_list()
    )
    out.loc[hybrid_mask, "mean_margin_minus_control_ttest_p_holm_hybrid"] = holm_adjust(
        out.loc[hybrid_mask, "mean_margin_minus_control_ttest_p"].to_list()
    )
    out.loc[hybrid_mask, "mean_margin_minus_control_wilcoxon_p_holm_hybrid"] = holm_adjust(
        out.loc[hybrid_mask, "mean_margin_minus_control_wilcoxon_p"].to_list()
    )
    return out


def build_hybrid_headline_table(model_summary: pd.DataFrame) -> pd.DataFrame:
    """Keep the hybrid model rows in a clean order for interpretation."""
    hybrid_df = model_summary.loc[model_summary["model_name"] == DEFAULT_HYBRID_MODEL].copy()
    if hybrid_df.empty:
        return hybrid_df

    region_order = {name: idx for idx, name in enumerate(DEFAULT_FOCUS_REGIONS)}
    window_order = {
        format_window_label(start_ms, end_ms): idx
        for idx, (start_ms, end_ms) in enumerate(DEFAULT_TARGET_WINDOWS_MS)
    }
    hybrid_df["region_order"] = hybrid_df["focus_region"].map(region_order)
    hybrid_df["window_order"] = hybrid_df["window_label"].map(window_order)
    hybrid_df = hybrid_df.sort_values(["region_order", "window_order"]).drop(
        columns=["region_order", "window_order"]
    )
    return hybrid_df.reset_index(drop=True)


def plot_hybrid_heatmaps(
    hybrid_df: pd.DataFrame,
    *,
    outfile: Path,
) -> None:
    """Plot the hybrid model's absolute advantage over size-matched controls."""
    if hybrid_df.empty:
        return

    region_order = list(DEFAULT_FOCUS_REGIONS)
    window_order = [format_window_label(start_ms, end_ms) for start_ms, end_ms in DEFAULT_TARGET_WINDOWS_MS]
    pair_matrix = (
        hybrid_df.pivot(index="focus_region", columns="window_label", values="pair_acc_minus_control_mean")
        .reindex(index=region_order, columns=window_order)
    )
    margin_matrix = (
        hybrid_df.pivot(index="focus_region", columns="window_label", values="mean_margin_minus_control_mean")
        .reindex(index=region_order, columns=window_order)
    )

    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    specs = [
        (
            axes[0],
            pair_matrix.to_numpy(dtype=float),
            "Hybrid pair-accuracy minus control",
            "Pair-accuracy delta",
        ),
        (
            axes[1],
            margin_matrix.to_numpy(dtype=float),
            "Hybrid score margin minus control",
            "Score-margin delta",
        ),
    ]

    for axis, matrix, title, cbar_label in specs:
        finite = matrix[np.isfinite(matrix)]
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_title(title)
        axis.set_xticks(np.arange(len(window_order)))
        axis.set_xticklabels(window_order)
        axis.set_yticks(np.arange(len(region_order)))
        axis.set_yticklabels(region_order)
        axis.set_xlabel("Target window (ms)")
        for row_idx, region_name in enumerate(region_order):
            for col_idx, window_label in enumerate(window_order):
                value = matrix[row_idx, col_idx]
                if np.isfinite(value):
                    axis.text(
                        col_idx,
                        row_idx,
                        f"{value:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#111111",
                    )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=cbar_label)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_sweep_report(
    hybrid_df: pd.DataFrame,
    gain_df: pd.DataFrame,
) -> str:
    """Write a short synthesis of the whole spatial-control sweep."""
    lines = [
        "# Matched Spatial-Control Sweep",
        "",
    ]
    if hybrid_df.empty:
        lines.append("No hybrid rows were available.")
        return "\n".join(lines)

    best_positive = hybrid_df.sort_values("pair_acc_minus_control_mean", ascending=False).iloc[0]
    best_negative = hybrid_df.sort_values("pair_acc_minus_control_mean", ascending=True).iloc[0]
    strongest_margin = hybrid_df.sort_values("mean_margin_minus_control_mean", ascending=False).iloc[0]
    weakest_margin = hybrid_df.sort_values("mean_margin_minus_control_mean", ascending=True).iloc[0]

    lines.extend(
        [
            "## Hybrid absolute performance versus controls",
            "",
            (
                f"- Strongest positive hybrid pair-accuracy advantage: {best_positive['focus_region']} "
                f"{best_positive['window_label']} ms, delta={best_positive['pair_acc_minus_control_mean']:.3f}, "
                f"paired t p={best_positive['pair_acc_minus_control_ttest_p']:.3g}, "
                f"Holm p={best_positive['pair_acc_minus_control_ttest_p_holm_hybrid']:.3g}."
            ),
            (
                f"- Strongest negative hybrid pair-accuracy contrast: {best_negative['focus_region']} "
                f"{best_negative['window_label']} ms, delta={best_negative['pair_acc_minus_control_mean']:.3f}, "
                f"paired t p={best_negative['pair_acc_minus_control_ttest_p']:.3g}, "
                f"Holm p={best_negative['pair_acc_minus_control_ttest_p_holm_hybrid']:.3g}."
            ),
            (
                f"- Strongest positive hybrid score-margin contrast: {strongest_margin['focus_region']} "
                f"{strongest_margin['window_label']} ms, delta={strongest_margin['mean_margin_minus_control_mean']:.3f}, "
                f"paired t p={strongest_margin['mean_margin_minus_control_ttest_p']:.3g}, "
                f"Holm p={strongest_margin['mean_margin_minus_control_ttest_p_holm_hybrid']:.3g}."
            ),
            (
                f"- Strongest negative hybrid score-margin contrast: {weakest_margin['focus_region']} "
                f"{weakest_margin['window_label']} ms, delta={weakest_margin['mean_margin_minus_control_mean']:.3f}, "
                f"paired t p={weakest_margin['mean_margin_minus_control_ttest_p']:.3g}, "
                f"Holm p={weakest_margin['mean_margin_minus_control_ttest_p_holm_hybrid']:.3g}."
            ),
            "",
        ]
    )

    if not gain_df.empty:
        best_gain = gain_df.sort_values("pair_acc_gain_minus_control_mean", ascending=False).iloc[0]
        worst_gain = gain_df.sort_values("pair_acc_gain_minus_control_mean", ascending=True).iloc[0]
        lines.extend(
            [
                "## Hybrid gain versus controls",
                "",
                (
                    f"- Largest positive hybrid-gain contrast: {best_gain['focus_region']} "
                    f"{best_gain['window_label']} ms, delta={best_gain['pair_acc_gain_minus_control_mean']:.3f}, "
                    f"paired t p={best_gain['pair_acc_gain_minus_control_ttest_p']:.3g}."
                ),
                (
                    f"- Largest negative hybrid-gain contrast: {worst_gain['focus_region']} "
                    f"{worst_gain['window_label']} ms, delta={worst_gain['pair_acc_gain_minus_control_mean']:.3f}, "
                    f"paired t p={worst_gain['pair_acc_gain_minus_control_ttest_p']:.3g}."
                ),
                "",
            ]
        )

    return "\n".join(lines)


def run_matched_spatial_control_sweep(
    *,
    baseline_results_dir: Path = DEFAULT_BASELINE_RESULTS_DIR,
    derivatives_dir: Path = DEFAULT_DERIVATIVES_DIR,
    export_dir: Path = DEFAULT_EXPORT_DIR,
    outdir: Path | None = None,
    rep_dim: int = DEFAULT_REP_DIM,
    focus_regions: tuple[str, ...] = DEFAULT_FOCUS_REGIONS,
    target_windows_ms: tuple[tuple[float, float], ...] = DEFAULT_TARGET_WINDOWS_MS,
    positive_sdt: str = DEFAULT_POSITIVE_SDT,
    negative_sdt: str = DEFAULT_NEGATIVE_SDT,
    random_seed: int = DEFAULT_RANDOM_SEED,
    n_bootstrap: int = DEFAULT_N_BOOT,
) -> int:
    """Run the full matched spatial-control sweep and collate the outputs."""
    outdir = (
        baseline_results_dir / "summary_matched_spatial_control_sweep"
        if outdir is None
        else outdir
    )
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        model_summary, gain_summary = collect_sweep_outputs(
            baseline_results_dir=baseline_results_dir,
            derivatives_dir=derivatives_dir,
            export_dir=export_dir,
            outdir=outdir,
            rep_dim=rep_dim,
            focus_regions=focus_regions,
            target_windows_ms=target_windows_ms,
            positive_sdt=positive_sdt,
            negative_sdt=negative_sdt,
            random_seed=random_seed,
            n_bootstrap=n_bootstrap,
        )
        model_summary = apply_hybrid_family_corrections(model_summary)
        hybrid_summary = build_hybrid_headline_table(model_summary)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    model_summary_path = outdir / "matched_spatial_control_sweep_model_summary.csv"
    gain_summary_path = outdir / "matched_spatial_control_sweep_gain_summary.csv"
    hybrid_summary_path = outdir / "matched_spatial_control_sweep_hybrid_summary.csv"
    plot_path = outdir / "matched_spatial_control_sweep_hybrid_heatmaps.png"
    report_path = outdir / "matched_spatial_control_sweep_summary.md"

    save_table(model_summary, model_summary_path)
    save_table(gain_summary, gain_summary_path)
    save_table(hybrid_summary, hybrid_summary_path)
    plot_hybrid_heatmaps(hybrid_summary, outfile=plot_path)
    report_path.write_text(
        build_sweep_report(hybrid_summary, gain_summary) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {model_summary_path}")
    print(f"Saved {gain_summary_path}")
    print(f"Saved {hybrid_summary_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {report_path}")
    return 0
