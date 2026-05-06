"""Build the central EEG VAR time-course figure from saved summaries."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_SUMMARY_DIR = Path(
    "data/results_baseline/region_sliding_baseline300ms_controls_focus_q4/"
    "summary_temporal_evidence_central"
)

MODEL_LABELS = {
    "Raw central+frontal delta": "Raw EEG",
    "Baseline central+frontal augmented": "VAR features",
    "Raw + baseline augmented": "Raw EEG + VAR",
    "Raw central delta": "Raw EEG",
    "Baseline central augmented": "VAR features",
    "Raw + baseline augmented (central)": "Raw EEG + VAR",
}

MODEL_COLOURS = {
    "Raw EEG": "#56616c",
    "VAR features": "#2f7f7f",
    "Raw EEG + VAR": "#d56a1c",
}


def _ci_half_width(std: float, n: int) -> float:
    if n < 2 or not np.isfinite(std):
        return math.nan
    return float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))


def _clean_group_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    df = group_summary.copy()
    df["plot_label"] = df["model_label"].map(MODEL_LABELS)
    df = df.dropna(subset=["plot_label"])
    if "target_center_ms" not in df.columns:
        df["target_center_ms"] = (df["target_start_ms"] + df["target_end_ms"]) / 2.0
    df["auc_ci95"] = [
        _ci_half_width(std, int(n))
        for std, n in zip(df["auc_std"], df["n_subjects_auc"], strict=True)
    ]
    df["rho_ci95"] = [
        _ci_half_width(std, int(n))
        for std, n in zip(df["hit_conf_rho_std"], df["n_subjects_hit_conf"], strict=True)
    ]
    return df.sort_values(["plot_label", "target_center_ms"])


def _format_window(start: float, end: float) -> str:
    return f"{int(start)}-{int(end)}"


def _plot_metric(
    axis,
    df: pd.DataFrame,
    *,
    metric: str,
    ci_col: str,
    ylabel: str,
    panel_label: str,
    chance_line: float,
) -> None:
    for label in ("Raw EEG", "VAR features", "Raw EEG + VAR"):
        model_df = df.loc[df["plot_label"].eq(label)].sort_values("target_center_ms")
        x = model_df["target_center_ms"].to_numpy(dtype=float)
        y = model_df[metric].to_numpy(dtype=float)
        colour = MODEL_COLOURS[label]
        axis.plot(x, y, marker="o", markersize=4.2, linewidth=1.7, color=colour, label=label)

    axis.axhline(chance_line, color="#707780", linewidth=0.9, linestyle="--")
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Post-stimulus window (ms)")
    axis.grid(axis="y", color="#e8eaed", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.text(
        -0.10,
        1.04,
        panel_label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def _write_caption(df: pd.DataFrame, pairwise: pd.DataFrame, outpath: Path) -> None:
    auc_rows = pairwise.loc[pairwise["metric"].eq("auc")].copy()
    preferred_pairs = (
        ("raw_regions_delta", "raw_plus_baseline_regions_augmented", 1.0),
        ("raw_plus_baseline_regions_augmented", "raw_regions_delta", -1.0),
        ("raw_cf_delta", "raw_plus_baseline_cf_augmented", 1.0),
        ("raw_plus_baseline_cf_augmented", "raw_cf_delta", -1.0),
    )
    auc_pair = pd.DataFrame()
    sign = 1.0
    for model_a, model_b, direction in preferred_pairs:
        candidate = auc_rows.loc[auc_rows["model_a"].eq(model_a) & auc_rows["model_b"].eq(model_b)]
        if not candidate.empty:
            auc_pair = candidate.copy()
            sign = direction
            break
    if auc_pair.empty:
        auc_pair = auc_rows.copy()
    auc_pair["window"] = [
        _format_window(s, e)
        for s, e in zip(auc_pair["target_start_ms"], auc_pair["target_end_ms"], strict=True)
    ]
    auc_pair["raw_plus_var_gain"] = sign * auc_pair["mean_delta_model_b_minus_a"]
    best = auc_pair.sort_values("raw_plus_var_gain", ascending=False).iloc[0]
    lines = [
        "# EEG VAR Time-Course Figure",
        "",
        "Central EEG evidence sweep using the PCA+VAR predictive-fit front end.",
        f"The largest raw-plus-VAR AUC gain over raw EEG is in the {best['window']} ms window: "
        f"delta={best['raw_plus_var_gain']:.3f}, paired p={best['ttest_p']:.3g}.",
        "",
        "The figure is descriptive context for choosing the focus cell; the main PRISM result is "
        "reported separately with subject-level PRISM statistics.",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def run(*, summary_dir: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    group_path = summary_dir / "temporal_evidence_group_summary.csv"
    pairwise_path = summary_dir / "temporal_evidence_pairwise_summary.csv"
    if not group_path.exists():
        group_path = summary_dir / "hybrid_evidence_group_summary.csv"
    if not pairwise_path.exists():
        pairwise_path = summary_dir / "hybrid_evidence_pairwise_summary.csv"
    group_summary = pd.read_csv(group_path)
    pairwise = pd.read_csv(pairwise_path)
    df = _clean_group_summary(group_summary)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism-mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "prism-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75), constrained_layout=True)
    _plot_metric(
        axes[0],
        df,
        metric="auc_mean",
        ci_col="auc_ci95",
        ylabel="Subject mean AUC",
        panel_label="A",
        chance_line=0.5,
    )
    _plot_metric(
        axes[1],
        df,
        metric="hit_conf_rho_mean",
        ci_col="rho_ci95",
        ylabel="Hit-trial confidence rho",
        panel_label="B",
        chance_line=0.0,
    )

    windows = (
        df[["target_start_ms", "target_end_ms", "target_center_ms"]]
        .drop_duplicates()
        .sort_values("target_center_ms")
    )
    ticks = windows["target_center_ms"].to_numpy(dtype=float)
    labels = [_format_window(s, e) for s, e in zip(windows["target_start_ms"], windows["target_end_ms"])]
    for axis in axes:
        axis.set_xticks(ticks, labels)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    fig.savefig(outdir / "eeg_var_timecourse.png", dpi=300)
    fig.savefig(outdir / "eeg_var_timecourse.pdf")
    plt.close(fig)

    df.to_csv(outdir / "eeg_var_timecourse_points.csv", index=False)
    _write_caption(df, pairwise, outdir / "eeg_var_timecourse_caption.md")
    print(f"Wrote {outdir / 'eeg_var_timecourse.pdf'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--outdir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary_dir = Path(args.summary_dir)
    outdir = summary_dir / "paper_figure" if args.outdir is None else Path(args.outdir)
    run(summary_dir=summary_dir, outdir=outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
