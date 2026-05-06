"""Subject-wise decoder summaries for central PRISM time windows."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

import pandas as pd

from eeg_subject_decoder import (
    DEFAULT_BASELINE_END_MS,
    DEFAULT_BASELINE_RESULTS,
    DEFAULT_BASELINE_START_MS,
    DEFAULT_DERIVATIVES_DIR,
    DEFAULT_EXPORT_DIR,
    DEFAULT_N_FOLDS,
    DEFAULT_PRISM_MODEL_FAMILY,
    DEFAULT_REGION,
    DEFAULT_REP_DIM,
    MODEL_SPECS,
    _ci95,
    _format_p,
    build_analysis_frame,
    decode_all_subjects,
    summarise_models,
    summarise_pairwise,
)


DEFAULT_ROOT = Path("data/results_prism/eeg_prism_central_timecourse_pca_q4")
DEFAULT_OUTDIR = DEFAULT_ROOT / "summary_subject_decoder_timecourse"
WINDOW_RE = re.compile(r"^window_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")


def _window_label(start_ms: float, end_ms: float) -> str:
    return f"{start_ms:g}-{end_ms:g}"


def _parse_window_dir(path: Path) -> tuple[float, float]:
    match = WINDOW_RE.match(path.name)
    if match is None:
        raise ValueError(f"Window directory must be named like window_125_375: {path}")
    return float(match.group(1)), float(match.group(2))


def discover_window_dirs(root: Path) -> list[tuple[Path, float, float]]:
    windows: list[tuple[Path, float, float]] = []
    for path in sorted(root.glob("window_*")):
        if not path.is_dir():
            continue
        start_ms, end_ms = _parse_window_dir(path)
        windows.append((path, start_ms, end_ms))
    if not windows:
        raise FileNotFoundError(f"No window_* directories found under {root}")
    return sorted(windows, key=lambda item: (item[1], item[2]))


def _add_window_columns(df: pd.DataFrame, *, start_ms: float, end_ms: float) -> pd.DataFrame:
    out = df.copy()
    out["target_start_ms"] = float(start_ms)
    out["target_end_ms"] = float(end_ms)
    out["target_center_ms"] = 0.5 * (float(start_ms) + float(end_ms))
    out["window_label"] = _window_label(start_ms, end_ms)
    return out


def _save_combined_prism_csv(window_dir: Path) -> None:
    files = sorted(window_dir.glob("sub-*_region_window_prism.csv"))
    if not files:
        return
    df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    sort_cols = [
        col
        for col in ("subject", "trial_idx", "region_name", "model_family", "rep_dim")
        if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols)
    df.to_csv(window_dir / "all_subjects_region_window_prism.csv", index=False)


def run_window(
    *,
    window_dir: Path,
    start_ms: float,
    end_ms: float,
    baseline_results_dir: Path,
    derivatives_dir: Path,
    export_dir: Path,
    region: str,
    rep_dim: int,
    baseline_start_ms: float,
    baseline_end_ms: float,
    prism_model_family: str,
    positive_sdt: str,
    negative_sdt: str,
    n_folds: int,
    ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _save_combined_prism_csv(window_dir)
    frame = build_analysis_frame(
        baseline_results_dir=baseline_results_dir,
        prism_results_dir=window_dir,
        derivatives_dir=derivatives_dir,
        export_dir=export_dir,
        region=region,
        rep_dim=rep_dim,
        test_start_ms=start_ms,
        test_end_ms=end_ms,
        baseline_start_ms=baseline_start_ms,
        baseline_end_ms=baseline_end_ms,
        prism_model_family=prism_model_family,
        positive_sdt=positive_sdt,
        negative_sdt=negative_sdt,
        n_folds=n_folds,
    )
    scores, subject_summary = decode_all_subjects(
        frame,
        model_specs=MODEL_SPECS,
        n_folds=n_folds,
        ridge=ridge,
    )
    group_summary = summarise_models(subject_summary)
    pairwise = summarise_pairwise(subject_summary)
    return (
        _add_window_columns(scores, start_ms=start_ms, end_ms=end_ms),
        _add_window_columns(subject_summary, start_ms=start_ms, end_ms=end_ms),
        _add_window_columns(group_summary, start_ms=start_ms, end_ms=end_ms),
        _add_window_columns(pairwise, start_ms=start_ms, end_ms=end_ms),
    )


def _plot_timecourse(group_summary: pd.DataFrame, pairwise: pd.DataFrame, outdir: Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism-mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "prism-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short_labels = {
        "raw_central_delta": "Raw EEG",
        "var_central_augmented": "VAR features",
        "prism_central_augmented": "PRISM summaries",
        "raw_plus_var": "Raw EEG + VAR",
        "raw_plus_prism": "Raw EEG + PRISM",
    }
    colours = {
        "raw_central_delta": "#56616c",
        "var_central_augmented": "#2f7f7f",
        "prism_central_augmented": "#7b6aa7",
        "raw_plus_var": "#d56a1c",
        "raw_plus_prism": "#6f9e75",
        "prism_gain_over_raw": "#6f9e75",
        "var_gain_over_raw": "#d56a1c",
        "var_minus_prism_hybrid": "#555555",
    }

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)

    ax = axes[0]
    for spec in MODEL_SPECS:
        model_df = group_summary.loc[group_summary["model_name"].eq(spec.name)].sort_values(
            "target_center_ms"
        )
        x = model_df["target_center_ms"].to_numpy(dtype=float)
        y = model_df["auc_mean"].to_numpy(dtype=float)
        low = model_df["auc_ci95_low"].to_numpy(dtype=float)
        high = model_df["auc_ci95_high"].to_numpy(dtype=float)
        colour = colours[spec.name]
        ax.plot(x, y, marker="o", markersize=4.0, linewidth=1.6, color=colour, label=short_labels[spec.name])
        ax.fill_between(x, low, high, color=colour, alpha=0.10, linewidth=0)
    ax.axhline(0.5, color="#707780", linewidth=0.9, linestyle="--")
    ax.set_ylabel("Held-out AUC")
    ax.set_xlabel("Post-stimulus window (ms)")
    ax.set_title("Subject-wise decoder")
    ax.grid(axis="y", color="#e8eaed", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.10, 1.04, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = axes[1]
    for contrast, label in (
        ("prism_gain_over_raw", "PRISM gain vs raw"),
        ("var_gain_over_raw", "VAR gain vs raw"),
        ("var_minus_prism_hybrid", "VAR hybrid minus PRISM hybrid"),
    ):
        contrast_df = pairwise.loc[pairwise["contrast"].eq(contrast)].sort_values("target_center_ms")
        x = contrast_df["target_center_ms"].to_numpy(dtype=float)
        y = contrast_df["mean_delta_model_b_minus_a"].to_numpy(dtype=float)
        low = contrast_df["ci95_low"].to_numpy(dtype=float)
        high = contrast_df["ci95_high"].to_numpy(dtype=float)
        colour = colours[contrast]
        ax.plot(x, y, marker="o", markersize=4.0, linewidth=1.6, color=colour, label=label)
        ax.fill_between(x, low, high, color=colour, alpha=0.10, linewidth=0)
    ax.axhline(0.0, color="#707780", linewidth=0.9, linestyle="--")
    ax.set_ylabel("Delta AUC")
    ax.set_xlabel("Post-stimulus window (ms)")
    ax.set_title("Paired subject-level contrasts")
    ax.grid(axis="y", color="#e8eaed", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.10, 1.04, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")

    windows = (
        group_summary[["target_start_ms", "target_end_ms", "target_center_ms"]]
        .drop_duplicates()
        .sort_values("target_center_ms")
    )
    ticks = windows["target_center_ms"].to_numpy(dtype=float)
    labels = [_window_label(s, e) for s, e in zip(windows["target_start_ms"], windows["target_end_ms"])]
    for axis in axes:
        axis.set_xticks(ticks, labels)
    axes[0].legend(frameon=False, fontsize=7.2, loc="upper left")
    axes[1].legend(frameon=False, fontsize=7.2, loc="upper left")

    fig.savefig(outdir / "eeg_prism_timecourse_decoder.png", dpi=300)
    fig.savefig(outdir / "eeg_prism_timecourse_decoder.pdf")
    plt.close(fig)


def _write_report(group_summary: pd.DataFrame, pairwise: pd.DataFrame, outpath: Path) -> None:
    lines = [
        "# EEG PRISM Timecourse Decoder",
        "",
        "Each window is decoded within subject, then tested across subjects.",
        "",
        "## AUC by window",
        "",
    ]
    for _, row in group_summary.sort_values(["target_center_ms", "model_name"]).iterrows():
        lines.append(
            f"- {row['window_label']} ms, {row['model_name']}: mean AUC={row['auc_mean']:.4f}, "
            f"95% CI [{row['auc_ci95_low']:.4f}, {row['auc_ci95_high']:.4f}], "
            f"p={_format_p(float(row['auc_ttest_vs_half_p']))}."
        )
    lines.extend(["", "## Main contrasts", ""])
    focus = pairwise.loc[
        pairwise["contrast"].isin(["prism_gain_over_raw", "var_gain_over_raw", "var_minus_prism_hybrid"])
    ].copy()
    for _, row in focus.sort_values(["target_center_ms", "contrast"]).iterrows():
        lines.append(
            f"- {row['window_label']} ms, {row['contrast']}: delta={row['mean_delta_model_b_minus_a']:.4f}, "
            f"95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], "
            f"{int(row['positive_subjects'])}/{int(row['n_subjects'])} positive subjects, "
            f"paired t p={_format_p(float(row['ttest_p']))}."
        )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    root: Path,
    baseline_results_dir: Path,
    derivatives_dir: Path,
    export_dir: Path,
    outdir: Path,
    region: str,
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
    scores_parts: list[pd.DataFrame] = []
    subject_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    pairwise_parts: list[pd.DataFrame] = []

    for window_dir, start_ms, end_ms in discover_window_dirs(root):
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
        scores_parts.append(scores)
        subject_parts.append(subject_summary)
        group_parts.append(group_summary)
        pairwise_parts.append(pairwise)

    scores_all = pd.concat(scores_parts, ignore_index=True)
    subject_all = pd.concat(subject_parts, ignore_index=True)
    group_all = pd.concat(group_parts, ignore_index=True)
    pairwise_all = pd.concat(pairwise_parts, ignore_index=True)

    scores_all.to_csv(outdir / "eeg_prism_timecourse_trial_scores.csv", index=False)
    subject_all.to_csv(outdir / "eeg_prism_timecourse_subject_summary.csv", index=False)
    group_all.to_csv(outdir / "eeg_prism_timecourse_group_summary.csv", index=False)
    pairwise_all.to_csv(outdir / "eeg_prism_timecourse_pairwise.csv", index=False)
    _plot_timecourse(group_all, pairwise_all, outdir)
    _write_report(group_all, pairwise_all, outdir / "eeg_prism_timecourse_decoder_summary.md")
    print(f"Wrote {outdir / 'eeg_prism_timecourse_group_summary.csv'}")
    print(f"Wrote {outdir / 'eeg_prism_timecourse_pairwise.csv'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline-results-dir", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--derivatives-dir", type=Path, default=DEFAULT_DERIVATIVES_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--region", default=DEFAULT_REGION)
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
        region=args.region,
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
