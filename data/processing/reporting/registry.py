"""Registry of result-summary tasks for the EEG processing pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from experiments.temporal import DEFAULT_STIM_LOCKED_WINDOWS
from framework.summaries import (
    compute_metric_means,
    compute_metric_summary,
    load_result_csvs,
    prepare_pyplot,
    safe_mode_int,
    save_table,
    select_best_rows,
)
from reporting.behaviour import DEFAULT_DERIVATIVES_DIR, run_behaviour_summary
from reporting.evidence import (
    DEFAULT_EXPORT_DIR,
    DEFAULT_NEGATIVE_SDT,
    DEFAULT_POSITIVE_SDT,
    DEFAULT_REGIONS,
    run_evidence_summary,
)
from reporting.matched_stimamp_evidence import (
    DEFAULT_CONFIDENCE_MIN_PER_SIDE,
    DEFAULT_CONFIDENCE_MIN_UNIQUE,
    DEFAULT_CONFIDENCE_SDT_SUBSET,
    DEFAULT_CONFIDENCE_TRIAL_SCORES_CSV,
    DEFAULT_DETECTION_FOCUS_CONTRAST_CSV,
    DEFAULT_FOCUS_END_MS as DEFAULT_MATCHED_FOCUS_END_MS,
    DEFAULT_FOCUS_MODEL_A as DEFAULT_MATCHED_FOCUS_MODEL_A,
    DEFAULT_FOCUS_MODEL_B as DEFAULT_MATCHED_FOCUS_MODEL_B,
    DEFAULT_FOCUS_START_MS as DEFAULT_MATCHED_FOCUS_START_MS,
    DEFAULT_MATCHED_TRIAL_SCORES_CSV,
    DEFAULT_MODEL_NAMES as DEFAULT_MATCHED_MODEL_NAMES,
    run_confidence_dissociation_summary,
    run_matched_stimamp_evidence_summary,
)
from reporting.matched_spatial_control import (
    DEFAULT_BASELINE_RESULTS_DIR as DEFAULT_MATCHED_SPATIAL_BASELINE_RESULTS_DIR,
    DEFAULT_FOCUS_REGION as DEFAULT_MATCHED_SPATIAL_FOCUS_REGION,
    DEFAULT_REP_DIM as DEFAULT_MATCHED_SPATIAL_REP_DIM,
    DEFAULT_TARGET_END_MS as DEFAULT_MATCHED_SPATIAL_TARGET_END_MS,
    DEFAULT_TARGET_START_MS as DEFAULT_MATCHED_SPATIAL_TARGET_START_MS,
    run_matched_spatial_control_summary,
)
from reporting.matched_spatial_control_sweep import (
    DEFAULT_FOCUS_REGIONS as DEFAULT_MATCHED_SPATIAL_SWEEP_REGIONS,
    DEFAULT_TARGET_WINDOWS_MS as DEFAULT_MATCHED_SPATIAL_SWEEP_WINDOWS,
    run_matched_spatial_control_sweep,
)
from reporting.mechanistic_dissociation import (
    DEFAULT_BASELINE_AUGMENTED_MODEL as DEFAULT_MECH_BASELINE_AUG_MODEL,
    DEFAULT_FOCUS_END_MS as DEFAULT_MECH_FOCUS_END_MS,
    DEFAULT_FOCUS_START_MS as DEFAULT_MECH_FOCUS_START_MS,
    DEFAULT_HYBRID_MODEL as DEFAULT_MECH_HYBRID_MODEL,
    DEFAULT_RAW_MODEL as DEFAULT_MECH_RAW_MODEL,
    run_mechanistic_dissociation_summary,
)
from reporting.spatial_dissociation import (
    DEFAULT_MATCHED_SPATIAL_SWEEP_CSV as DEFAULT_SPATIAL_DISSOCIATION_SWEEP_CSV,
    DEFAULT_REGION_CONTROL_SUMMARY_CSV as DEFAULT_SPATIAL_DISSOCIATION_CONTROL_CSV,
    run_spatial_dissociation_summary,
)
from reporting.spatial import run_region_sliding_summary
from reporting.statistics import (
    DEFAULT_N_BOOT,
    DEFAULT_RANDOM_SEED,
    run_baseline_statistics,
    run_context_history_summary,
    run_control_summary,
    run_prism_summary,
    run_temporal_context_summary,
)
from reporting.stimamp_evidence import (
    DEFAULT_FOCUS_END_MS,
    DEFAULT_FOCUS_MODEL_A,
    DEFAULT_FOCUS_MODEL_B,
    DEFAULT_FOCUS_START_MS,
    DEFAULT_MODEL_NAMES as DEFAULT_STIMAMP_MODEL_NAMES,
    DEFAULT_STIMAMP_BIN_LABELS,
    DEFAULT_TEMPORAL_TRIAL_SCORES_CSV,
    run_stimamp_evidence_summary,
)
from reporting.temporal_evidence import (
    DEFAULT_TEMPORAL_EVIDENCE_RESULTS_DIR,
    run_temporal_evidence_summary,
)


ParserMutator = Callable[[argparse.ArgumentParser], None]
SummaryRunner = Callable[[argparse.Namespace], int]


WINDOW_ORDER = [name for name, _, _ in DEFAULT_STIM_LOCKED_WINDOWS]


@dataclass(frozen=True)
class SummaryTaskSpec:
    name: str
    description: str
    add_arguments: ParserMutator
    run: SummaryRunner
    aliases: tuple[str, ...] = ()


def add_results_dir_subject_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_results_dir: str,
    results_help: str,
) -> None:
    parser.add_argument(
        "--results-dir",
        default=default_results_dir,
        help=results_help,
    )
    parser.add_argument(
        "--subject",
        default="sub-01",
        help="Subject id for the single-subject summary outputs.",
    )


def load_baseline_results(results_dir: Path) -> pd.DataFrame:
    return load_result_csvs(
        results_dir=results_dir,
        pattern="sub-*_trial_baseline.csv",
        required_columns=[
            "subject",
            "trial_idx",
            "rep_dim",
            "pred_mse_obs",
            "pred_r2_obs",
            "pred_mse_latent",
            "pred_r2_latent",
            "pred_nll_latent",
        ],
        sort_columns=["subject", "trial_idx", "rep_dim"],
    )


def summarise_subject(df: pd.DataFrame, subject: str, *, group_columns: list[str]) -> pd.DataFrame:
    subject_df = df.loc[df["subject"] == subject]
    if subject_df.empty:
        raise ValueError(f"No rows found for subject {subject}")

    return compute_metric_summary(subject_df, group_columns=group_columns)


def summarise_subject_means(
    df: pd.DataFrame,
    *,
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_means = compute_metric_means(df, group_columns=["subject", *group_columns])
    group_summary = compute_metric_summary(subject_means, group_columns=group_columns)
    return subject_means, group_summary


def plot_baseline_subject_summary(
    summary_df: pd.DataFrame,
    subject: str,
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    rep_dim = summary_df["rep_dim"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].errorbar(
        rep_dim,
        summary_df["pred_r2_obs_mean"],
        yerr=summary_df["pred_r2_obs_std"],
        fmt="-o",
        color="#0f4c5c",
        capsize=3,
        linewidth=2,
    )
    axes[0].set_title(f"{subject} mean observed R^2")
    axes[0].set_xlabel("Representation dimension")
    axes[0].set_ylabel("pred_r2_obs")
    axes[0].grid(True, alpha=0.25)

    axes[1].errorbar(
        rep_dim,
        summary_df["pred_mse_obs_mean"],
        yerr=summary_df["pred_mse_obs_std"],
        fmt="-o",
        color="#e36414",
        capsize=3,
        linewidth=2,
    )
    axes[1].set_title(f"{subject} mean observed MSE")
    axes[1].set_xlabel("Representation dimension")
    axes[1].set_ylabel("pred_mse_obs")
    axes[1].grid(True, alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_group_summary(
    subject_means: pd.DataFrame,
    group_summary: pd.DataFrame,
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for _, subject_df in subject_means.groupby("subject"):
        axes[0].plot(
            subject_df["rep_dim"],
            subject_df["pred_r2_obs"],
            color="#6c757d",
            alpha=0.25,
            linewidth=1,
        )
        axes[1].plot(
            subject_df["rep_dim"],
            subject_df["pred_mse_obs"],
            color="#6c757d",
            alpha=0.25,
            linewidth=1,
        )

    axes[0].errorbar(
        group_summary["rep_dim"],
        group_summary["pred_r2_obs_mean"],
        yerr=group_summary["pred_r2_obs_std"],
        fmt="-o",
        color="#0f4c5c",
        capsize=3,
        linewidth=2.5,
    )
    axes[0].set_title("Across subjects mean observed R^2")
    axes[0].set_xlabel("Representation dimension")
    axes[0].set_ylabel("pred_r2_obs")
    axes[0].grid(True, alpha=0.25)

    axes[1].errorbar(
        group_summary["rep_dim"],
        group_summary["pred_mse_obs_mean"],
        yerr=group_summary["pred_mse_obs_std"],
        fmt="-o",
        color="#e36414",
        capsize=3,
        linewidth=2.5,
    )
    axes[1].set_title("Across subjects mean observed MSE")
    axes[1].set_xlabel("Representation dimension")
    axes[1].set_ylabel("pred_mse_obs")
    axes[1].grid(True, alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_baseline_summary(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)

    try:
        all_df = load_baseline_results(results_dir)
        subject_summary = summarise_subject(all_df, args.subject, group_columns=["rep_dim"])
        subject_means, group_summary = summarise_subject_means(
            all_df,
            group_columns=["rep_dim"],
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    subject_table = results_dir / f"{args.subject}_summary_by_rep_dim.csv"
    subject_plot = results_dir / f"{args.subject}_summary_plots.png"
    subject_means_table = results_dir / "all_subjects_subject_mean_by_rep_dim.csv"
    group_table = results_dir / "all_subjects_group_summary_by_rep_dim.csv"
    group_plot = results_dir / "all_subjects_summary_plots.png"

    save_table(subject_summary, subject_table)
    save_table(subject_means, subject_means_table)
    save_table(group_summary, group_table)
    plot_baseline_subject_summary(subject_summary, args.subject, subject_plot)
    plot_baseline_group_summary(subject_means, group_summary, group_plot)

    print(f"Saved {subject_table}")
    print(f"Saved {subject_plot}")
    print(f"Saved {subject_means_table}")
    print(f"Saved {group_table}")
    print(f"Saved {group_plot}")

    print("\nSubject summary:")
    print(subject_summary.to_string(index=False))

    print("\nAcross-subject summary:")
    print(group_summary.to_string(index=False))
    return 0


def load_windowed_results(results_dir: Path) -> pd.DataFrame:
    return load_result_csvs(
        results_dir=results_dir,
        pattern="sub-*_windowed_baseline.csv",
        required_columns=[
            "subject",
            "trial_idx",
            "window_name",
            "rep_dim",
            "pred_mse_obs",
            "pred_r2_obs",
        ],
        sort_columns=["subject", "trial_idx", "window_name", "rep_dim"],
        categorical_orders={"window_name": WINDOW_ORDER},
    )


def plot_windowed_subject_summary(summary_df: pd.DataFrame, subject: str, outfile: Path) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    for window_name in WINDOW_ORDER:
        window_df = summary_df.loc[summary_df["window_name"] == window_name]
        if window_df.empty:
            continue
        axes[0].plot(
            window_df["rep_dim"],
            window_df["pred_r2_obs_mean"],
            marker="o",
            linewidth=2,
            label=window_name,
        )
        axes[1].plot(
            window_df["rep_dim"],
            window_df["pred_mse_obs_mean"],
            marker="o",
            linewidth=2,
            label=window_name,
        )

    axes[0].set_title(f"{subject} windowed mean observed R^2")
    axes[0].set_xlabel("Representation dimension")
    axes[0].set_ylabel("pred_r2_obs")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title(f"{subject} windowed mean observed MSE")
    axes[1].set_xlabel("Representation dimension")
    axes[1].set_ylabel("pred_mse_obs")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_windowed_group_summary(group_summary: pd.DataFrame, outfile: Path) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    for window_name in WINDOW_ORDER:
        window_df = group_summary.loc[group_summary["window_name"] == window_name]
        if window_df.empty:
            continue
        axes[0].plot(
            window_df["rep_dim"],
            window_df["pred_r2_obs_mean"],
            marker="o",
            linewidth=2,
            label=window_name,
        )
        axes[1].plot(
            window_df["rep_dim"],
            window_df["pred_mse_obs_mean"],
            marker="o",
            linewidth=2,
            label=window_name,
        )

    axes[0].set_title("Across subjects mean observed R^2 by window")
    axes[0].set_xlabel("Representation dimension")
    axes[0].set_ylabel("pred_r2_obs")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title("Across subjects mean observed MSE by window")
    axes[1].set_xlabel("Representation dimension")
    axes[1].set_ylabel("pred_mse_obs")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_windowed_summary(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)

    try:
        all_df = load_windowed_results(results_dir)
        subject_summary = summarise_subject(
            all_df,
            args.subject,
            group_columns=["window_name", "rep_dim"],
        )
        subject_means, group_summary = summarise_subject_means(
            all_df,
            group_columns=["window_name", "rep_dim"],
        )
        best_q_df = select_best_rows(
            subject_means,
            group_columns=["subject", "window_name"],
            score_column="pred_r2_obs",
            value_columns=["rep_dim", "pred_r2_obs", "pred_mse_obs"],
            sort_columns=["window_name", "subject"],
        ).rename(
            columns={
                "rep_dim": "best_rep_dim",
                "pred_r2_obs": "best_pred_r2_obs",
                "pred_mse_obs": "best_pred_mse_obs",
            }
        )
        best_q_counts = (
            best_q_df.groupby(["window_name", "best_rep_dim"])
            .size()
            .rename("subject_count")
            .reset_index()
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    subject_table = results_dir / f"{args.subject}_windowed_summary_by_rep_dim.csv"
    subject_plot = results_dir / f"{args.subject}_windowed_summary_plots.png"
    subject_means_table = results_dir / "all_subjects_windowed_subject_mean_by_rep_dim.csv"
    group_table = results_dir / "all_subjects_windowed_group_summary_by_rep_dim.csv"
    group_plot = results_dir / "all_subjects_windowed_summary_plots.png"
    best_q_table = results_dir / "all_subjects_windowed_best_q_by_subject.csv"
    best_q_count_table = results_dir / "all_subjects_windowed_best_q_counts.csv"

    save_table(subject_summary, subject_table)
    save_table(subject_means, subject_means_table)
    save_table(group_summary, group_table)
    save_table(best_q_df, best_q_table)
    save_table(best_q_counts, best_q_count_table)
    plot_windowed_subject_summary(subject_summary, args.subject, subject_plot)
    plot_windowed_group_summary(group_summary, group_plot)

    print(f"Saved {subject_table}")
    print(f"Saved {subject_plot}")
    print(f"Saved {subject_means_table}")
    print(f"Saved {group_table}")
    print(f"Saved {group_plot}")
    print(f"Saved {best_q_table}")
    print(f"Saved {best_q_count_table}")

    print("\nAcross-subject windowed summary:")
    print(group_summary.to_string(index=False))

    print("\nBest q counts by window:")
    print(best_q_counts.to_string(index=False))
    return 0


def load_multiscale_results(results_dir: Path) -> pd.DataFrame:
    return load_result_csvs(
        results_dir=results_dir,
        pattern="sub-*_multiscale_baseline.csv",
        required_columns=[
            "subject",
            "trial_idx",
            "scale_name",
            "duration_ms",
            "rep_dim",
            "pred_mse_obs",
            "pred_r2_obs",
        ],
        sort_columns=["subject", "trial_idx", "duration_ms", "rep_dim"],
    )


def plot_multiscale_subject_summary(summary_df: pd.DataFrame, subject: str, outfile: Path) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    for rep_dim in sorted(summary_df["rep_dim"].unique()):
        rep_df = summary_df.loc[summary_df["rep_dim"] == rep_dim]
        axes[0].plot(
            rep_df["duration_ms"],
            rep_df["pred_r2_obs_mean"],
            marker="o",
            linewidth=2,
            label=f"q={rep_dim}",
        )
        axes[1].plot(
            rep_df["duration_ms"],
            rep_df["pred_mse_obs_mean"],
            marker="o",
            linewidth=2,
            label=f"q={rep_dim}",
        )

    axes[0].set_title(f"{subject} mean observed R^2 by duration")
    axes[0].set_xlabel("Duration (ms)")
    axes[0].set_ylabel("pred_r2_obs")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title(f"{subject} mean observed MSE by duration")
    axes[1].set_xlabel("Duration (ms)")
    axes[1].set_ylabel("pred_mse_obs")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multiscale_group_summary(
    group_summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for rep_dim in sorted(group_summary["rep_dim"].unique()):
        rep_df = group_summary.loc[group_summary["rep_dim"] == rep_dim]
        axes[0].plot(
            rep_df["duration_ms"],
            rep_df["pred_r2_obs_mean"],
            marker="o",
            linewidth=2,
            label=f"q={rep_dim}",
        )
        axes[1].plot(
            rep_df["duration_ms"],
            rep_df["pred_mse_obs_mean"],
            marker="o",
            linewidth=2,
            label=f"q={rep_dim}",
        )

    axes[2].errorbar(
        best_summary["duration_ms"],
        best_summary["best_pred_r2_obs_mean"],
        yerr=best_summary["best_pred_r2_obs_std"],
        fmt="-o",
        color="#0f4c5c",
        linewidth=2.5,
        capsize=3,
    )

    axes[0].set_title("Across subjects mean observed R^2")
    axes[0].set_xlabel("Duration (ms)")
    axes[0].set_ylabel("pred_r2_obs")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title("Across subjects mean observed MSE")
    axes[1].set_xlabel("Duration (ms)")
    axes[1].set_ylabel("pred_mse_obs")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    axes[2].set_title("Best over q: across-subject mean observed R^2")
    axes[2].set_xlabel("Duration (ms)")
    axes[2].set_ylabel("best_pred_r2_obs")
    axes[2].grid(True, alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_multiscale_summary(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)

    try:
        all_df = load_multiscale_results(results_dir)
        subject_summary = summarise_subject(
            all_df,
            args.subject,
            group_columns=["duration_ms", "rep_dim"],
        )
        subject_means, group_summary = summarise_subject_means(
            all_df,
            group_columns=["duration_ms", "rep_dim"],
        )
        best_df = select_best_rows(
            subject_means,
            group_columns=["subject", "duration_ms"],
            score_column="pred_r2_obs",
            value_columns=["rep_dim", "pred_r2_obs", "pred_mse_obs"],
            sort_columns=["duration_ms", "subject"],
        ).rename(
            columns={
                "rep_dim": "best_rep_dim",
                "pred_r2_obs": "best_pred_r2_obs",
                "pred_mse_obs": "best_pred_mse_obs",
            }
        )
        best_summary = compute_metric_summary(
            best_df,
            group_columns=["duration_ms"],
            metric_columns=["best_pred_r2_obs", "best_pred_mse_obs"],
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    subject_table = results_dir / f"{args.subject}_multiscale_summary_by_rep_dim.csv"
    subject_plot = results_dir / f"{args.subject}_multiscale_summary_plots.png"
    subject_means_table = results_dir / "all_subjects_multiscale_subject_mean_by_rep_dim.csv"
    group_table = results_dir / "all_subjects_multiscale_group_summary_by_rep_dim.csv"
    best_table = results_dir / "all_subjects_multiscale_best_over_q_by_subject.csv"
    best_summary_table = results_dir / "all_subjects_multiscale_best_over_q_summary.csv"
    group_plot = results_dir / "all_subjects_multiscale_summary_plots.png"

    save_table(subject_summary, subject_table)
    save_table(subject_means, subject_means_table)
    save_table(group_summary, group_table)
    save_table(best_df, best_table)
    save_table(best_summary, best_summary_table)
    plot_multiscale_subject_summary(subject_summary, args.subject, subject_plot)
    plot_multiscale_group_summary(group_summary, best_summary, group_plot)

    print(f"Saved {subject_table}")
    print(f"Saved {subject_plot}")
    print(f"Saved {subject_means_table}")
    print(f"Saved {group_table}")
    print(f"Saved {best_table}")
    print(f"Saved {best_summary_table}")
    print(f"Saved {group_plot}")

    print("\nAcross-subject multiscale summary:")
    print(group_summary.to_string(index=False))

    print("\nBest-over-q summary by duration:")
    print(best_summary.to_string(index=False))
    return 0


def load_context_sweep_results(results_dir: Path) -> pd.DataFrame:
    return load_result_csvs(
        results_dir=results_dir,
        pattern="sub-*_context_sweep_baseline.csv",
        required_columns=[
            "subject",
            "trial_idx",
            "history_ms",
            "target_start_ms",
            "target_end_ms",
            "rep_dim",
            "pred_mse_obs",
            "pred_r2_obs",
        ],
        sort_columns=["subject", "trial_idx", "history_ms", "rep_dim"],
    )


def plot_context_sweep_subject_summary(
    summary_df: pd.DataFrame,
    subject: str,
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    for rep_dim in sorted(summary_df["rep_dim"].unique()):
        rep_df = summary_df.loc[summary_df["rep_dim"] == rep_dim]
        axes[0].plot(
            rep_df["history_ms"],
            rep_df["pred_r2_obs_mean"],
            marker="o",
            linewidth=2,
            label=f"q={rep_dim}",
        )
        axes[1].plot(
            rep_df["history_ms"],
            rep_df["pred_mse_obs_mean"],
            marker="o",
            linewidth=2,
            label=f"q={rep_dim}",
        )

    axes[0].set_title(f"{subject} mean observed R^2 by history")
    axes[0].set_xlabel("History duration (ms)")
    axes[0].set_ylabel("pred_r2_obs")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title(f"{subject} mean observed MSE by history")
    axes[1].set_xlabel("History duration (ms)")
    axes[1].set_ylabel("pred_mse_obs")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_context_sweep_group_summary(
    group_summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for rep_dim in sorted(group_summary["rep_dim"].unique()):
        rep_df = group_summary.loc[group_summary["rep_dim"] == rep_dim]
        axes[0].plot(
            rep_df["history_ms"],
            rep_df["pred_r2_obs_mean"],
            marker="o",
            linewidth=2,
            label=f"q={rep_dim}",
        )
        axes[1].plot(
            rep_df["history_ms"],
            rep_df["pred_mse_obs_mean"],
            marker="o",
            linewidth=2,
            label=f"q={rep_dim}",
        )

    axes[2].errorbar(
        best_summary["history_ms"],
        best_summary["best_pred_r2_obs_mean"],
        yerr=best_summary["best_pred_r2_obs_std"],
        fmt="-o",
        color="#0f4c5c",
        linewidth=2.5,
        capsize=3,
    )

    axes[0].set_title("Across subjects mean observed R^2")
    axes[0].set_xlabel("History duration (ms)")
    axes[0].set_ylabel("pred_r2_obs")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title("Across subjects mean observed MSE")
    axes[1].set_xlabel("History duration (ms)")
    axes[1].set_ylabel("pred_mse_obs")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    axes[2].set_title("Best over q: across-subject mean observed R^2")
    axes[2].set_xlabel("History duration (ms)")
    axes[2].set_ylabel("best_pred_r2_obs")
    axes[2].grid(True, alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_context_sweep_summary(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)

    try:
        all_df = load_context_sweep_results(results_dir)
        subject_summary = summarise_subject(
            all_df,
            args.subject,
            group_columns=["history_ms", "rep_dim"],
        )
        subject_means, group_summary = summarise_subject_means(
            all_df,
            group_columns=["history_ms", "rep_dim"],
        )
        best_df = select_best_rows(
            subject_means,
            group_columns=["subject", "history_ms"],
            score_column="pred_r2_obs",
            value_columns=["rep_dim", "pred_r2_obs", "pred_mse_obs"],
            sort_columns=["history_ms", "subject"],
        ).rename(
            columns={
                "rep_dim": "best_rep_dim",
                "pred_r2_obs": "best_pred_r2_obs",
                "pred_mse_obs": "best_pred_mse_obs",
            }
        )
        best_summary = compute_metric_summary(
            best_df,
            group_columns=["history_ms"],
            metric_columns=["best_pred_r2_obs", "best_pred_mse_obs"],
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    subject_table = results_dir / f"{args.subject}_context_sweep_summary_by_rep_dim.csv"
    subject_plot = results_dir / f"{args.subject}_context_sweep_summary_plots.png"
    subject_means_table = results_dir / "all_subjects_context_sweep_subject_mean_by_rep_dim.csv"
    group_table = results_dir / "all_subjects_context_sweep_group_summary_by_rep_dim.csv"
    best_table = results_dir / "all_subjects_context_sweep_best_over_q_by_subject.csv"
    best_summary_table = results_dir / "all_subjects_context_sweep_best_over_q_summary.csv"
    group_plot = results_dir / "all_subjects_context_sweep_summary_plots.png"

    save_table(subject_summary, subject_table)
    save_table(subject_means, subject_means_table)
    save_table(group_summary, group_table)
    save_table(best_df, best_table)
    save_table(best_summary, best_summary_table)
    plot_context_sweep_subject_summary(subject_summary, args.subject, subject_plot)
    plot_context_sweep_group_summary(group_summary, best_summary, group_plot)

    print(f"Saved {subject_table}")
    print(f"Saved {subject_plot}")
    print(f"Saved {subject_means_table}")
    print(f"Saved {group_table}")
    print(f"Saved {best_table}")
    print(f"Saved {best_summary_table}")
    print(f"Saved {group_plot}")

    print("\nAcross-subject context sweep summary:")
    print(group_summary.to_string(index=False))

    print("\nBest-over-q summary by history duration:")
    print(best_summary.to_string(index=False))
    return 0


def load_centre_sweep_results(results_dir: Path) -> pd.DataFrame:
    return load_result_csvs(
        results_dir=results_dir,
        pattern="sub-*_center_sweep_baseline.csv",
        required_columns=[
            "subject",
            "trial_idx",
            "duration_ms",
            "center_ms",
            "rep_dim",
            "pred_mse_obs",
            "pred_r2_obs",
        ],
        sort_columns=["subject", "trial_idx", "duration_ms", "center_ms", "rep_dim"],
    )


def pivot_metric(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return df.pivot(index="duration_ms", columns="center_ms", values=value_col).sort_index()


def draw_heatmap(
    ax,
    matrix: pd.DataFrame,
    *,
    plt,
    title: str,
    cmap: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Centre (ms)")
    ax.set_ylabel("Duration (ms)")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([f"{int(round(x))}" for x in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([f"{int(round(y))}" for y in matrix.index])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def plot_centre_sweep_subject_summary(
    best_over_q_df: pd.DataFrame,
    subject_best_center_df: pd.DataFrame,
    subject: str,
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    subject_best = best_over_q_df.loc[best_over_q_df["subject"] == subject]
    if subject_best.empty:
        raise ValueError(f"No best-over-q rows found for subject {subject}")

    r2_matrix = pivot_metric(subject_best, "best_pred_r2_obs")
    q_matrix = pivot_metric(subject_best, "best_rep_dim")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    draw_heatmap(
        axes[0],
        r2_matrix,
        plt=plt,
        title=f"{subject}: best-over-q observed R^2",
        cmap="viridis",
        cbar_label="best_pred_r2_obs",
    )
    draw_heatmap(
        axes[1],
        q_matrix,
        plt=plt,
        title=f"{subject}: best q by duration and centre",
        cmap="cividis",
        cbar_label="best_rep_dim",
        vmin=2,
        vmax=32,
    )

    axes[2].plot(
        subject_best_center_df["duration_ms"],
        subject_best_center_df["best_pred_r2_obs"],
        marker="o",
        linewidth=2,
        color="#0f4c5c",
    )
    axes[2].set_title(f"{subject}: best over centre and q")
    axes[2].set_xlabel("Duration (ms)")
    axes[2].set_ylabel("best_pred_r2_obs")
    axes[2].grid(True, alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_centre_sweep_group_summary(
    best_summary: pd.DataFrame,
    best_center_summary: pd.DataFrame,
    *,
    n_subjects: int,
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    r2_matrix = pivot_metric(best_summary, "best_pred_r2_obs_mean")
    positive_matrix = pivot_metric(best_summary, "positive_subjects")
    q16_matrix = pivot_metric(best_summary, "q16_share")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    draw_heatmap(
        axes[0],
        r2_matrix,
        plt=plt,
        title="Across subjects mean best-over-q observed R^2",
        cmap="viridis",
        cbar_label="mean best_pred_r2_obs",
    )
    draw_heatmap(
        axes[1],
        positive_matrix,
        plt=plt,
        title="Positive-subject count at best q",
        cmap="magma",
        cbar_label="subjects with R^2 > 0",
        vmin=0,
        vmax=max(1, n_subjects),
    )
    draw_heatmap(
        axes[2],
        q16_matrix,
        plt=plt,
        title="Share of subjects with q=16 as best q",
        cmap="plasma",
        cbar_label="q16_share",
        vmin=0.0,
        vmax=1.0,
    )

    axes[3].errorbar(
        best_center_summary["duration_ms"],
        best_center_summary["best_pred_r2_obs_mean"],
        yerr=best_center_summary["best_pred_r2_obs_std"],
        fmt="-o",
        linewidth=2.5,
        capsize=3,
        color="#0f4c5c",
    )
    axes[3].set_title("Best over centre and q by duration")
    axes[3].set_xlabel("Duration (ms)")
    axes[3].set_ylabel("best_pred_r2_obs")
    axes[3].grid(True, alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_centre_sweep_summary(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)

    try:
        all_df = load_centre_sweep_results(results_dir)
        n_subjects = int(all_df["subject"].nunique())
        subject_summary = summarise_subject(
            all_df,
            args.subject,
            group_columns=["duration_ms", "center_ms", "rep_dim"],
        )
        subject_means, group_summary = summarise_subject_means(
            all_df,
            group_columns=["duration_ms", "center_ms", "rep_dim"],
        )
        best_df = select_best_rows(
            subject_means,
            group_columns=["subject", "duration_ms", "center_ms"],
            score_column="pred_r2_obs",
            value_columns=["rep_dim", "pred_r2_obs", "pred_mse_obs"],
            allow_missing_score=True,
            sort_columns=["duration_ms", "center_ms", "subject"],
        ).rename(
            columns={
                "rep_dim": "best_rep_dim",
                "pred_r2_obs": "best_pred_r2_obs",
                "pred_mse_obs": "best_pred_mse_obs",
            }
        )
        grouped_best = best_df.groupby(["duration_ms", "center_ms"], observed=False)
        best_summary = compute_metric_summary(
            best_df,
            group_columns=["duration_ms", "center_ms"],
            metric_columns=["best_pred_r2_obs", "best_pred_mse_obs"],
        )
        best_summary["positive_subjects"] = grouped_best["best_pred_r2_obs"].apply(
            lambda s: int((s.dropna() > 0).sum())
        ).to_numpy()
        best_summary["available_subjects"] = grouped_best["best_pred_r2_obs"].apply(
            lambda s: int(s.notna().sum())
        ).to_numpy()
        best_summary["q16_share"] = grouped_best["best_rep_dim"].apply(
            lambda s: float(np.mean(s.dropna() == 16)) if s.notna().any() else np.nan
        ).to_numpy()
        best_summary["best_rep_dim_mode"] = grouped_best["best_rep_dim"].apply(
            safe_mode_int
        ).to_numpy()

        best_center_df = select_best_rows(
            subject_means,
            group_columns=["subject", "duration_ms"],
            score_column="pred_r2_obs",
            value_columns=["center_ms", "rep_dim", "pred_r2_obs", "pred_mse_obs"],
            allow_missing_score=True,
            sort_columns=["duration_ms", "subject"],
        ).rename(
            columns={
                "center_ms": "best_center_ms",
                "rep_dim": "best_rep_dim",
                "pred_r2_obs": "best_pred_r2_obs",
                "pred_mse_obs": "best_pred_mse_obs",
            }
        )
        grouped_centre = best_center_df.groupby("duration_ms", observed=False)
        best_center_summary = compute_metric_summary(
            best_center_df,
            group_columns=["duration_ms"],
            metric_columns=["best_pred_r2_obs", "best_pred_mse_obs", "best_center_ms"],
        )
        best_center_summary["n_best_center_within_250ms"] = grouped_centre[
            "best_center_ms"
        ].apply(lambda s: int((s.dropna().abs() <= 250.0).sum())).to_numpy()
        best_center_summary["best_rep_dim_mode"] = grouped_centre["best_rep_dim"].apply(
            safe_mode_int
        ).to_numpy()
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    subject_table = results_dir / f"{args.subject}_center_sweep_summary_by_rep_dim.csv"
    subject_plot = results_dir / f"{args.subject}_center_sweep_summary_plots.png"
    subject_means_table = results_dir / "all_subjects_center_sweep_subject_mean_by_rep_dim.csv"
    group_table = results_dir / "all_subjects_center_sweep_group_summary_by_rep_dim.csv"
    best_table = results_dir / "all_subjects_center_sweep_best_over_q_by_subject.csv"
    best_summary_table = results_dir / "all_subjects_center_sweep_best_over_q_summary.csv"
    best_center_table = results_dir / "all_subjects_center_sweep_best_over_center_and_q_by_subject.csv"
    best_center_summary_table = (
        results_dir / "all_subjects_center_sweep_best_over_center_and_q_summary.csv"
    )
    group_plot = results_dir / "all_subjects_center_sweep_summary_plots.png"

    save_table(subject_summary, subject_table)
    save_table(subject_means, subject_means_table)
    save_table(group_summary, group_table)
    save_table(best_df, best_table)
    save_table(best_summary, best_summary_table)
    save_table(best_center_df, best_center_table)
    save_table(best_center_summary, best_center_summary_table)
    plot_centre_sweep_subject_summary(
        best_df,
        best_center_df.loc[best_center_df["subject"] == args.subject],
        args.subject,
        subject_plot,
    )
    plot_centre_sweep_group_summary(
        best_summary,
        best_center_summary,
        n_subjects=n_subjects,
        outfile=group_plot,
    )

    print(f"Loaded {n_subjects} subject file(s) from {results_dir}")
    print(f"Saved {subject_table}")
    print(f"Saved {subject_plot}")
    print(f"Saved {subject_means_table}")
    print(f"Saved {group_table}")
    print(f"Saved {best_table}")
    print(f"Saved {best_summary_table}")
    print(f"Saved {best_center_table}")
    print(f"Saved {best_center_summary_table}")
    print(f"Saved {group_plot}")

    print("\nAcross-subject centre-sweep summary:")
    print(best_summary.to_string(index=False))

    print("\nBest over centre and q by duration:")
    print(best_center_summary.to_string(index=False))
    return 0


def add_baseline_summary_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the plain baseline summary."""
    add_results_dir_subject_arguments(
        parser,
        default_results_dir="./data/results_baseline",
        results_help="Directory containing per-subject baseline CSVs.",
    )


def add_windowed_summary_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the windowed baseline summary."""
    add_results_dir_subject_arguments(
        parser,
        default_results_dir="./data/results_baseline/windowed",
        results_help="Directory containing per-subject windowed baseline CSVs.",
    )


def add_multiscale_summary_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the multiscale baseline summary."""
    add_results_dir_subject_arguments(
        parser,
        default_results_dir="./data/results_baseline/multiscale",
        results_help="Directory containing per-subject multiscale baseline CSVs.",
    )


def add_context_sweep_summary_arguments(parser: argparse.ArgumentParser) -> None:
    add_results_dir_subject_arguments(
        parser,
        default_results_dir="./data/results_baseline/context_sweep",
        results_help="Directory containing per-subject context sweep baseline CSVs.",
    )


def add_centre_sweep_summary_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the centre-sweep baseline summary."""
    add_results_dir_subject_arguments(
        parser,
        default_results_dir="./data/results_baseline/center_sweep",
        results_help="Directory containing per-subject centre-sweep baseline CSVs.",
    )


def add_statistics_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the baseline statistics task."""
    parser.add_argument(
        "--results-dir",
        default="./data/results_baseline",
        help="Directory containing per-subject baseline CSVs.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory for the statistics tables and plots. Defaults to <results-dir>/statistics.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for bootstrap resampling.",
    )


def add_control_summary_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the control-comparison task."""
    parser.add_argument(
        "--results-dir",
        default="./data/results_baseline",
        help="Root baseline results directory containing the controls/ folder.",
    )
    parser.add_argument(
        "--controls-dir",
        default=None,
        help="Directory containing the control CSVs. Defaults to <results-dir>/controls.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for bootstrap resampling.",
    )


def add_behaviour_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the behaviour-link summary task."""
    parser.add_argument(
        "--baseline-csv",
        default="./data/results_baseline/all_subjects_trial_baseline.csv",
        help="Path to the combined all-subjects baseline CSV.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default=str(DEFAULT_DERIVATIVES_DIR),
        help="Root directory containing EEGPREP per-subject .set files.",
    )
    parser.add_argument(
        "--outdir",
        default="./data/results_baseline/behaviour",
        help="Directory for merged outputs and summary tables.",
    )


def add_prism_summary_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the PRISM comparison summary."""
    parser.add_argument(
        "--baseline-results-dir",
        default="./data/results_baseline",
        help="Directory containing the existing baseline per-subject CSVs.",
    )
    parser.add_argument(
        "--results-dir",
        default="./data/results_prism",
        help="Directory containing the PRISM per-subject CSVs.",
    )
    parser.add_argument(
        "--subject",
        default="sub-01",
        help="Subject id for the single-subject comparison table and plot.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the summary products.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for bootstrap resampling.",
    )


def add_temporal_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-dir",
        default="./data/results_baseline",
        help="Root baseline results directory containing the multiscale outputs.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the temporal-context summary products.",
    )


def add_context_history_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--free-results-dir",
        default="./data/results_baseline/context_sweep",
        help="Directory containing the unrestricted context sweep summaries.",
    )
    parser.add_argument(
        "--matched-results-dir",
        default="./data/results_baseline/context_sweep_matched_pairs",
        help="Directory containing the matched-pair context sweep summaries.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the context-history comparison products.",
    )


def add_region_sliding_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-dir",
        default="./data/results_baseline/region_sliding",
        help="Directory containing the region-sliding CSVs.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default=str(DEFAULT_DERIVATIVES_DIR),
        help="Root directory containing EEGPREP per-subject .set files.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the region-sliding summary products.",
    )
    parser.add_argument(
        "--rep-dim",
        type=int,
        default=None,
        help="Optional representation dimension filter for the summary tables and plots.",
    )


def add_evidence_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline-results-dir",
        default="./data/results_baseline/region_window_cf_q4_pilot10",
        help="Directory containing the regional baseline CSVs for the evidence analysis.",
    )
    parser.add_argument(
        "--prism-results-dir",
        default="./data/results_prism/region_window_cf_q4_pilot10_pca",
        help="Directory containing the regional PRISM CSVs for the evidence analysis.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default=str(DEFAULT_DERIVATIVES_DIR),
        help="Root directory containing EEGPREP per-subject .set files.",
    )
    parser.add_argument(
        "--export-dir",
        default=str(DEFAULT_EXPORT_DIR),
        help="Directory containing the exported MATLAB trial files used for raw EEG evidence features.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the evidence summary products.",
    )
    parser.add_argument(
        "--rep-dim",
        type=int,
        default=4,
        help="Representation dimension to include in the evidence features.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGIONS),
        help="Regions to use when constructing the evidence feature sets.",
    )
    parser.add_argument(
        "--test-start-ms",
        type=float,
        default=125.0,
        help="Start of the post-stimulus target window in milliseconds.",
    )
    parser.add_argument(
        "--test-end-ms",
        type=float,
        default=375.0,
        help="End of the post-stimulus target window in milliseconds.",
    )
    parser.add_argument(
        "--positive-sdt",
        default=DEFAULT_POSITIVE_SDT,
        help="Positive SDT label used for the evidence discriminant.",
    )
    parser.add_argument(
        "--negative-sdt",
        default=DEFAULT_NEGATIVE_SDT,
        help="Negative SDT label used for the evidence discriminant.",
    )
    parser.add_argument(
        "--prism-model-family",
        default="prism_pca",
        help="PRISM model family to use when building the evidence features.",
    )


def add_temporal_evidence_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline-results-dir",
        default=str(DEFAULT_TEMPORAL_EVIDENCE_RESULTS_DIR),
        help="Directory containing the fixed-baseline regional baseline CSVs for the temporal evidence sweep.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default=str(DEFAULT_DERIVATIVES_DIR),
        help="Root directory containing EEGPREP per-subject .set files.",
    )
    parser.add_argument(
        "--export-dir",
        default=str(DEFAULT_EXPORT_DIR),
        help="Directory containing the exported MATLAB trial files used for raw EEG features.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the temporal evidence summary products.",
    )
    parser.add_argument(
        "--rep-dim",
        type=int,
        default=4,
        help="Representation dimension to include from the baseline results.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGIONS),
        help="Regions to combine for the raw, augmented, and hybrid evidence models.",
    )
    parser.add_argument(
        "--positive-sdt",
        default=DEFAULT_POSITIVE_SDT,
        help="Positive SDT label used for the evidence discriminant.",
    )
    parser.add_argument(
        "--negative-sdt",
        default=DEFAULT_NEGATIVE_SDT,
        help="Negative SDT label used for the evidence discriminant.",
    )


def add_stimamp_evidence_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trial-scores-csv",
        default=str(DEFAULT_TEMPORAL_TRIAL_SCORES_CSV),
        help="Temporal evidence trial-score CSV to stratify by stimulus amplitude.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default=str(DEFAULT_DERIVATIVES_DIR),
        help="Root directory containing EEGPREP per-subject .set files.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the stimulus-amplitude evidence summary products.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=list(DEFAULT_STIMAMP_MODEL_NAMES),
        help="Evidence models to keep when building the stimulus-amplitude summary.",
    )
    parser.add_argument(
        "--positive-sdt",
        default=DEFAULT_POSITIVE_SDT,
        help="Positive SDT label used for the evidence discriminant.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=len(DEFAULT_STIMAMP_BIN_LABELS),
        help="Number of within-subject stimulus-amplitude bins.",
    )
    parser.add_argument(
        "--bin-labels",
        nargs="+",
        default=list(DEFAULT_STIMAMP_BIN_LABELS),
        help="Labels used for the stimulus-amplitude bins.",
    )
    parser.add_argument(
        "--focus-model-a",
        default=DEFAULT_FOCUS_MODEL_A,
        help="Reference model for the pairwise contrast.",
    )
    parser.add_argument(
        "--focus-model-b",
        default=DEFAULT_FOCUS_MODEL_B,
        help="Comparison model for the pairwise contrast.",
    )
    parser.add_argument(
        "--focus-start-ms",
        type=float,
        default=DEFAULT_FOCUS_START_MS,
        help="Preferred target-window start for the headline contrast.",
    )
    parser.add_argument(
        "--focus-end-ms",
        type=float,
        default=DEFAULT_FOCUS_END_MS,
        help="Preferred target-window end for the headline contrast.",
    )


def add_matched_stimamp_evidence_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trial-scores-csv",
        default=str(DEFAULT_MATCHED_TRIAL_SCORES_CSV),
        help="Evidence trial-score CSV to analyse with one-to-one stimulus-amplitude matching.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default=str(DEFAULT_DERIVATIVES_DIR),
        help="Root directory containing EEGPREP per-subject .set files.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the matched stimulus-amplitude summary products.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=list(DEFAULT_MATCHED_MODEL_NAMES),
        help="Evidence models to include in the matched stimulus-amplitude summary.",
    )
    parser.add_argument(
        "--positive-sdt",
        default=DEFAULT_POSITIVE_SDT,
        help="Positive SDT label used when forming the matched pairs.",
    )
    parser.add_argument(
        "--negative-sdt",
        default=DEFAULT_NEGATIVE_SDT,
        help="Negative SDT label used when forming the matched pairs.",
    )
    parser.add_argument(
        "--focus-model-a",
        default=DEFAULT_MATCHED_FOCUS_MODEL_A,
        help="Reference model for the pairwise matched-pair contrast.",
    )
    parser.add_argument(
        "--focus-model-b",
        default=DEFAULT_MATCHED_FOCUS_MODEL_B,
        help="Comparison model for the pairwise matched-pair contrast.",
    )
    parser.add_argument(
        "--focus-start-ms",
        type=float,
        default=DEFAULT_MATCHED_FOCUS_START_MS,
        help="Preferred target-window start for the headline matched-pair contrast.",
    )
    parser.add_argument(
        "--focus-end-ms",
        type=float,
        default=DEFAULT_MATCHED_FOCUS_END_MS,
        help="Preferred target-window end for the headline matched-pair contrast.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for bootstrap resampling.",
    )


def add_confidence_dissociation_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trial-scores-csv",
        default=str(DEFAULT_CONFIDENCE_TRIAL_SCORES_CSV),
        help=(
            "Trial-score CSV containing per-trial evidence, sdt and confidence columns. "
            "Defaults to the central 125-375 ms matched-spatial-control trial scores."
        ),
    )
    parser.add_argument(
        "--derivatives-dir",
        default=str(DEFAULT_DERIVATIVES_DIR),
        help="Root directory containing EEGPREP per-subject .set files (used if stimamp must be merged).",
    )
    parser.add_argument(
        "--detection-focus-contrasts-csv",
        default=str(DEFAULT_DETECTION_FOCUS_CONTRAST_CSV),
        help=(
            "Existing matched-stimamp focus-contrasts CSV to load alongside the new confidence "
            "contrasts so the dissociation can be reported side-by-side. Pass an empty string to skip."
        ),
    )
    parser.add_argument(
        "--detection-group-summary-csv",
        default="",
        help=(
            "Optional matched-stimamp group-summary CSV used by the dissociation plot to render the "
            "detection panel. Pass an empty string to skip the detection panel."
        ),
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the confidence dissociation summary products.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=list(DEFAULT_MATCHED_MODEL_NAMES),
        help="Evidence models to include in the confidence dissociation summary.",
    )
    parser.add_argument(
        "--sdt-subset",
        default=DEFAULT_CONFIDENCE_SDT_SUBSET,
        help="SDT subset to compute confidence pairs over (default: hit).",
    )
    parser.add_argument(
        "--min-unique-confidence",
        type=int,
        default=DEFAULT_CONFIDENCE_MIN_UNIQUE,
        help="Minimum number of unique confidence ratings per subject to keep that subject.",
    )
    parser.add_argument(
        "--min-per-side",
        type=int,
        default=DEFAULT_CONFIDENCE_MIN_PER_SIDE,
        help="Minimum trials on each side of the median confidence split.",
    )
    parser.add_argument(
        "--focus-model-a",
        default=DEFAULT_MATCHED_FOCUS_MODEL_A,
        help="Reference model for the pairwise confidence-pair contrast.",
    )
    parser.add_argument(
        "--focus-model-b",
        default=DEFAULT_MATCHED_FOCUS_MODEL_B,
        help="Comparison model for the pairwise confidence-pair contrast.",
    )
    parser.add_argument(
        "--focus-start-ms",
        type=float,
        default=DEFAULT_MATCHED_FOCUS_START_MS,
        help="Preferred target-window start for the headline confidence-pair contrast.",
    )
    parser.add_argument(
        "--focus-end-ms",
        type=float,
        default=DEFAULT_MATCHED_FOCUS_END_MS,
        help="Preferred target-window end for the headline confidence-pair contrast.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for bootstrap resampling.",
    )


def add_matched_spatial_control_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline-results-dir",
        default=str(DEFAULT_MATCHED_SPATIAL_BASELINE_RESULTS_DIR),
        help="Region-sliding baseline directory containing the named regions and size-matched controls.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default=str(DEFAULT_DERIVATIVES_DIR),
        help="Root directory containing EEGPREP per-subject .set files.",
    )
    parser.add_argument(
        "--export-dir",
        default=str(DEFAULT_EXPORT_DIR),
        help="Directory containing the MATLAB-exported EEG subject files.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the matched spatial-control summary products.",
    )
    parser.add_argument(
        "--rep-dim",
        type=int,
        default=DEFAULT_MATCHED_SPATIAL_REP_DIM,
        help="Representation dimension used in the baseline region-sliding run.",
    )
    parser.add_argument(
        "--focus-region",
        default=DEFAULT_MATCHED_SPATIAL_FOCUS_REGION,
        help="Named region to compare against its size-matched controls.",
    )
    parser.add_argument(
        "--target-start-ms",
        type=float,
        default=DEFAULT_MATCHED_SPATIAL_TARGET_START_MS,
        help="Target-window start in milliseconds.",
    )
    parser.add_argument(
        "--target-end-ms",
        type=float,
        default=DEFAULT_MATCHED_SPATIAL_TARGET_END_MS,
        help="Target-window end in milliseconds.",
    )
    parser.add_argument(
        "--positive-sdt",
        default=DEFAULT_POSITIVE_SDT,
        help="Positive SDT label used when forming the matched pairs.",
    )
    parser.add_argument(
        "--negative-sdt",
        default=DEFAULT_NEGATIVE_SDT,
        help="Negative SDT label used when forming the matched pairs.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Random seed used for confidence intervals and for rebuilding the deterministic "
            "size-matched control channel groups. This should match the experiment seed."
        ),
    )


def add_matched_spatial_control_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline-results-dir",
        default=str(DEFAULT_MATCHED_SPATIAL_BASELINE_RESULTS_DIR),
        help="Region-sliding baseline directory containing the named regions and size-matched controls.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default=str(DEFAULT_DERIVATIVES_DIR),
        help="Root directory containing EEGPREP per-subject .set files.",
    )
    parser.add_argument(
        "--export-dir",
        default=str(DEFAULT_EXPORT_DIR),
        help="Directory containing the MATLAB-exported EEG subject files.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the matched spatial-control sweep products.",
    )
    parser.add_argument(
        "--rep-dim",
        type=int,
        default=DEFAULT_MATCHED_SPATIAL_REP_DIM,
        help="Representation dimension used in the baseline region-sliding run.",
    )
    parser.add_argument(
        "--focus-regions",
        nargs="+",
        default=list(DEFAULT_MATCHED_SPATIAL_SWEEP_REGIONS),
        help="Named regions to include in the matched spatial-control sweep.",
    )
    parser.add_argument(
        "--target-windows-ms",
        nargs="+",
        type=float,
        default=[value for window in DEFAULT_MATCHED_SPATIAL_SWEEP_WINDOWS for value in window],
        help=(
            "Flat list of start/end millisecond pairs for the target windows. "
            "Example: --target-windows-ms 0 250 125 375 250 500"
        ),
    )
    parser.add_argument(
        "--positive-sdt",
        default=DEFAULT_POSITIVE_SDT,
        help="Positive SDT label used when forming the matched pairs.",
    )
    parser.add_argument(
        "--negative-sdt",
        default=DEFAULT_NEGATIVE_SDT,
        help="Negative SDT label used when forming the matched pairs.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for confidence intervals and the deterministic control groups.",
    )


def add_spatial_dissociation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--region-control-summary-csv",
        default=str(DEFAULT_SPATIAL_DISSOCIATION_CONTROL_CSV),
        help="Region-sliding named-versus-control summary CSV.",
    )
    parser.add_argument(
        "--matched-spatial-sweep-csv",
        default=str(DEFAULT_SPATIAL_DISSOCIATION_SWEEP_CSV),
        help="Hybrid matched spatial-control sweep summary CSV.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for the spatial dissociation summary.",
    )


def run_statistics_task(args: argparse.Namespace) -> int:
    """Dispatch the baseline statistics summary."""
    return run_baseline_statistics(
        results_dir=Path(args.results_dir),
        outdir=None if args.outdir is None else Path(args.outdir),
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
    )


def run_control_task(args: argparse.Namespace) -> int:
    """Dispatch the control-comparison summary."""
    return run_control_summary(
        results_dir=Path(args.results_dir),
        controls_dir=None if args.controls_dir is None else Path(args.controls_dir),
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
    )


def run_behaviour_task(args: argparse.Namespace) -> int:
    """Dispatch the behaviour-linked summary."""
    return run_behaviour_summary(
        baseline_csv=Path(args.baseline_csv),
        derivatives_dir=Path(args.derivatives_dir),
        outdir=Path(args.outdir),
    )


def run_prism_task(args: argparse.Namespace) -> int:
    """Dispatch the baseline-versus-PRISM summary."""
    return run_prism_summary(
        baseline_results_dir=Path(args.baseline_results_dir),
        prism_results_dir=Path(args.results_dir),
        outdir=None if args.outdir is None else Path(args.outdir),
        subject=args.subject,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
    )


def run_temporal_context_task(args: argparse.Namespace) -> int:
    return run_temporal_context_summary(
        results_dir=Path(args.results_dir),
        outdir=None if args.outdir is None else Path(args.outdir),
    )


def run_context_history_task(args: argparse.Namespace) -> int:
    return run_context_history_summary(
        free_results_dir=Path(args.free_results_dir),
        matched_results_dir=Path(args.matched_results_dir),
        outdir=None if args.outdir is None else Path(args.outdir),
    )


def run_region_sliding_task(args: argparse.Namespace) -> int:
    return run_region_sliding_summary(
        results_dir=Path(args.results_dir),
        derivatives_dir=Path(args.derivatives_dir),
        outdir=None if args.outdir is None else Path(args.outdir),
        rep_dim=args.rep_dim,
    )


def run_evidence_task(args: argparse.Namespace) -> int:
    return run_evidence_summary(
        baseline_results_dir=Path(args.baseline_results_dir),
        prism_results_dir=Path(args.prism_results_dir),
        derivatives_dir=Path(args.derivatives_dir),
        export_dir=Path(args.export_dir),
        outdir=None if args.outdir is None else Path(args.outdir),
        rep_dim=args.rep_dim,
        regions=tuple(args.regions),
        test_start_ms=args.test_start_ms,
        test_end_ms=args.test_end_ms,
        positive_sdt=args.positive_sdt,
        negative_sdt=args.negative_sdt,
        prism_model_family=args.prism_model_family,
    )


def run_temporal_evidence_task(args: argparse.Namespace) -> int:
    return run_temporal_evidence_summary(
        baseline_results_dir=Path(args.baseline_results_dir),
        derivatives_dir=Path(args.derivatives_dir),
        export_dir=Path(args.export_dir),
        outdir=None if args.outdir is None else Path(args.outdir),
        rep_dim=args.rep_dim,
        regions=tuple(args.regions),
        positive_sdt=args.positive_sdt,
        negative_sdt=args.negative_sdt,
    )


def run_stimamp_evidence_task(args: argparse.Namespace) -> int:
    return run_stimamp_evidence_summary(
        trial_scores_csv=Path(args.trial_scores_csv),
        derivatives_dir=Path(args.derivatives_dir),
        outdir=Path(args.outdir) if args.outdir else None,
        model_names=tuple(args.model_names),
        positive_sdt=args.positive_sdt,
        n_bins=int(args.n_bins),
        bin_labels=tuple(args.bin_labels),
        focus_model_a=args.focus_model_a,
        focus_model_b=args.focus_model_b,
        focus_start_ms=float(args.focus_start_ms),
        focus_end_ms=float(args.focus_end_ms),
    )


def run_matched_stimamp_evidence_task(args: argparse.Namespace) -> int:
    return run_matched_stimamp_evidence_summary(
        trial_scores_csv=Path(args.trial_scores_csv),
        derivatives_dir=Path(args.derivatives_dir),
        outdir=Path(args.outdir) if args.outdir else None,
        model_names=tuple(args.model_names),
        positive_sdt=args.positive_sdt,
        negative_sdt=args.negative_sdt,
        focus_model_a=args.focus_model_a,
        focus_model_b=args.focus_model_b,
        focus_start_ms=float(args.focus_start_ms),
        focus_end_ms=float(args.focus_end_ms),
        n_bootstrap=int(args.n_bootstrap),
        random_seed=int(args.random_seed),
    )


def run_confidence_dissociation_task(args: argparse.Namespace) -> int:
    detection_focus = (
        Path(args.detection_focus_contrasts_csv)
        if args.detection_focus_contrasts_csv
        else None
    )
    detection_group = (
        Path(args.detection_group_summary_csv)
        if args.detection_group_summary_csv
        else None
    )
    return run_confidence_dissociation_summary(
        trial_scores_csv=Path(args.trial_scores_csv),
        derivatives_dir=Path(args.derivatives_dir),
        detection_focus_contrasts_csv=detection_focus,
        detection_group_summary_csv=detection_group,
        outdir=Path(args.outdir) if args.outdir else None,
        model_names=tuple(args.model_names),
        sdt_subset=str(args.sdt_subset),
        min_unique_confidence=int(args.min_unique_confidence),
        min_per_side=int(args.min_per_side),
        focus_model_a=str(args.focus_model_a),
        focus_model_b=str(args.focus_model_b),
        focus_start_ms=float(args.focus_start_ms),
        focus_end_ms=float(args.focus_end_ms),
        n_bootstrap=int(args.n_bootstrap),
        random_seed=int(args.random_seed),
    )


def run_matched_spatial_control_task(args: argparse.Namespace) -> int:
    return run_matched_spatial_control_summary(
        baseline_results_dir=Path(args.baseline_results_dir),
        derivatives_dir=Path(args.derivatives_dir),
        export_dir=Path(args.export_dir),
        outdir=Path(args.outdir) if args.outdir else None,
        rep_dim=int(args.rep_dim),
        focus_region=str(args.focus_region),
        target_start_ms=float(args.target_start_ms),
        target_end_ms=float(args.target_end_ms),
        positive_sdt=str(args.positive_sdt),
        negative_sdt=str(args.negative_sdt),
        random_state=int(args.random_seed),
        n_bootstrap=int(args.n_bootstrap),
    )


def run_matched_spatial_control_sweep_task(args: argparse.Namespace) -> int:
    values = [float(value) for value in args.target_windows_ms]
    if len(values) % 2 != 0:
        raise ValueError("--target-windows-ms must contain start/end pairs")
    target_windows_ms = tuple(
        (values[idx], values[idx + 1])
        for idx in range(0, len(values), 2)
    )
    return run_matched_spatial_control_sweep(
        baseline_results_dir=Path(args.baseline_results_dir),
        derivatives_dir=Path(args.derivatives_dir),
        export_dir=Path(args.export_dir),
        outdir=Path(args.outdir) if args.outdir else None,
        rep_dim=int(args.rep_dim),
        focus_regions=tuple(str(region) for region in args.focus_regions),
        target_windows_ms=target_windows_ms,
        positive_sdt=str(args.positive_sdt),
        negative_sdt=str(args.negative_sdt),
        random_seed=int(args.random_seed),
        n_bootstrap=int(args.n_bootstrap),
    )


def run_spatial_dissociation_task(args: argparse.Namespace) -> int:
    return run_spatial_dissociation_summary(
        region_control_summary_csv=Path(args.region_control_summary_csv),
        matched_spatial_sweep_csv=Path(args.matched_spatial_sweep_csv),
        outdir=Path(args.outdir) if args.outdir else None,
    )


def add_mechanistic_dissociation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trial-scores-csv",
        required=True,
        help="Trial-score CSV with per-trial evidence_score across model_name values (e.g. central_125_375 matched_spatial_control_trial_scores.csv).",
    )
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--raw-model", default=DEFAULT_MECH_RAW_MODEL)
    parser.add_argument("--hybrid-model", default=DEFAULT_MECH_HYBRID_MODEL)
    parser.add_argument("--baseline-aug-model", default=DEFAULT_MECH_BASELINE_AUG_MODEL)
    parser.add_argument("--focus-start-ms", type=float, default=DEFAULT_MECH_FOCUS_START_MS)
    parser.add_argument("--focus-end-ms", type=float, default=DEFAULT_MECH_FOCUS_END_MS)


def run_mechanistic_dissociation_task(args: argparse.Namespace) -> int:
    return run_mechanistic_dissociation_summary(
        trial_scores_csv=Path(args.trial_scores_csv),
        outdir=Path(args.outdir) if args.outdir else None,
        raw_model=str(args.raw_model),
        hybrid_model=str(args.hybrid_model),
        baseline_aug_model=str(args.baseline_aug_model),
        focus_start_ms=float(args.focus_start_ms),
        focus_end_ms=float(args.focus_end_ms),
    )


SUMMARY_TASKS = (
    SummaryTaskSpec(
        name="baseline",
        description="Summarise the plain trial baseline outputs into tables and plots.",
        add_arguments=add_baseline_summary_arguments,
        run=run_baseline_summary,
    ),
    SummaryTaskSpec(
        name="windowed",
        description="Summarise the stimulus-locked windowed baseline outputs.",
        add_arguments=add_windowed_summary_arguments,
        run=run_windowed_summary,
    ),
    SummaryTaskSpec(
        name="multiscale",
        description="Summarise the centred multiscale baseline outputs.",
        add_arguments=add_multiscale_summary_arguments,
        run=run_multiscale_summary,
    ),
    SummaryTaskSpec(
        name="context-sweep",
        description="Summarise the fixed-target context sweep outputs.",
        add_arguments=add_context_sweep_summary_arguments,
        run=run_context_sweep_summary,
    ),
    SummaryTaskSpec(
        name="centre-sweep",
        description="Summarise the duration x centre sweep baseline outputs.",
        add_arguments=add_centre_sweep_summary_arguments,
        run=run_centre_sweep_summary,
        aliases=("center-sweep",),
    ),
    SummaryTaskSpec(
        name="statistics",
        description="Compute subject-level statistical summaries for the plain baseline.",
        add_arguments=add_statistics_arguments,
        run=run_statistics_task,
    ),
    SummaryTaskSpec(
        name="control",
        description="Summarise the baseline-versus-control comparison at subject level.",
        add_arguments=add_control_summary_arguments,
        run=run_control_task,
    ),
    SummaryTaskSpec(
        name="behaviour",
        description="Join baseline outputs to behaviour-linked epoch metadata.",
        add_arguments=add_behaviour_arguments,
        run=run_behaviour_task,
        aliases=("behavior",),
    ),
    SummaryTaskSpec(
        name="prism",
        description="Summarise the baseline-versus-PRISM comparison.",
        add_arguments=add_prism_summary_arguments,
        run=run_prism_task,
    ),
    SummaryTaskSpec(
        name="temporal-context",
        description="Summarise how predictive fit changes with temporal context.",
        add_arguments=add_temporal_context_arguments,
        run=run_temporal_context_task,
    ),
    SummaryTaskSpec(
        name="context-history",
        description="Compare unrestricted and matched-pair context sweeps.",
        add_arguments=add_context_history_arguments,
        run=run_context_history_task,
    ),
    SummaryTaskSpec(
        name="region-sliding",
        description="Summarise the regional sliding baseline, condition contrasts, and size-matched controls.",
        add_arguments=add_region_sliding_summary_arguments,
        run=run_region_sliding_task,
    ),
    SummaryTaskSpec(
        name="evidence",
        description="Run the held-out hit-versus-miss evidence analysis and confidence comparison.",
        add_arguments=add_evidence_summary_arguments,
        run=run_evidence_task,
    ),
    SummaryTaskSpec(
        name="temporal-evidence",
        description="Run the time-resolved raw versus augmented baseline evidence comparison.",
        add_arguments=add_temporal_evidence_summary_arguments,
        run=run_temporal_evidence_task,
    ),
    SummaryTaskSpec(
        name="stimamp-evidence",
        description="Stratify temporal evidence scores by within-subject stimulus-amplitude bins.",
        add_arguments=add_stimamp_evidence_summary_arguments,
        run=run_stimamp_evidence_task,
    ),
    SummaryTaskSpec(
        name="matched-stimamp-evidence",
        description="Analyse evidence scores after one-to-one within-subject stimulus-amplitude matching.",
        add_arguments=add_matched_stimamp_evidence_summary_arguments,
        run=run_matched_stimamp_evidence_task,
    ),
    SummaryTaskSpec(
        name="matched-spatial-control",
        description="Compare a named region against its size-matched controls using amplitude-matched evidence gains.",
        add_arguments=add_matched_spatial_control_summary_arguments,
        run=run_matched_spatial_control_task,
    ),
    SummaryTaskSpec(
        name="confidence-dissociation",
        description="Test detection-vs-confidence specificity by running matched-pair contrasts on within-subject high-vs-low confidence hits.",
        add_arguments=add_confidence_dissociation_summary_arguments,
        run=run_confidence_dissociation_task,
    ),
    SummaryTaskSpec(
        name="matched-spatial-control-sweep",
        description="Run the matched spatial-control comparison across all selected regions and windows.",
        add_arguments=add_matched_spatial_control_sweep_arguments,
        run=run_matched_spatial_control_sweep_task,
    ),
    SummaryTaskSpec(
        name="spatial-dissociation",
        description="Relate regional predictive-fit advantages to matched-evidence advantages.",
        add_arguments=add_spatial_dissociation_arguments,
        run=run_spatial_dissociation_task,
    ),
    SummaryTaskSpec(
        name="mechanistic-dissociation",
        description="Trial-level mixed-effects + per-subject Spearman partial correlation isolating the pre-stim VAR predictive-fit contribution to detection vs confidence.",
        add_arguments=add_mechanistic_dissociation_arguments,
        run=run_mechanistic_dissociation_task,
    ),
)


SUMMARY_INDEX = {
    name: spec
    for spec in SUMMARY_TASKS
    for name in (spec.name, *spec.aliases)
}


def get_summary_task(name: str) -> SummaryTaskSpec:
    """Look up a summary task by name or alias."""
    return SUMMARY_INDEX[name]
