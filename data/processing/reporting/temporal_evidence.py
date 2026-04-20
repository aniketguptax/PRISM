"""Time-resolved held-out evidence summaries for raw and predictive features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eegprep import DEFAULT_DERIVATIVES_DIR
from framework.summaries import prepare_pyplot, save_table
from reporting.evidence import (
    DEFAULT_EXPORT_DIR,
    DEFAULT_NEGATIVE_SDT,
    DEFAULT_POSITIVE_SDT,
    DEFAULT_REGIONS,
    assign_subject_twofold_splits,
    build_feature_frame,
    build_model_trial_table,
    build_pairwise_model_comparisons,
    build_raw_feature_frame,
    evaluate_subject_metrics,
    fit_out_of_fold_evidence,
    load_baseline_region_results,
    load_signal_metadata,
    merge_feature_frames,
    select_raw_feature_frame,
    summarise_model_metrics,
)


DEFAULT_TEMPORAL_EVIDENCE_RESULTS_DIR = Path(
    "./data/results_baseline/region_sliding_baseline300ms_controls_focus_q4"
)
DEFAULT_AUGMENTED_METRICS = (
    "pred_r2_obs",
    "pred_mse_obs",
    "pred_r2_latent",
    "pred_nll_latent",
)


def filter_temporal_baseline_rows(
    df: pd.DataFrame,
    *,
    rep_dim: int,
    regions: tuple[str, ...],
) -> pd.DataFrame:
    filtered = df.loc[
        (df["group_kind"] == "named_region")
        & (df["rep_dim"] == int(rep_dim))
        & (df["region_name"].isin(regions))
    ].copy()
    if filtered.empty:
        raise ValueError("No named-region baseline rows matched the requested temporal evidence setup")
    return filtered


def build_temporal_model_definitions(
    regions: tuple[str, ...],
) -> list[tuple[str, str, str, tuple[str, ...]]]:
    region_label = "+".join(regions)
    return [
        ("raw_regions_delta", f"Raw {region_label} delta", "raw", ()),
        ("baseline_regions_r2", f"Baseline {region_label} R^2", "baseline", ("pred_r2_obs",)),
        (
            "baseline_regions_augmented",
            f"Baseline {region_label} augmented",
            "baseline",
            DEFAULT_AUGMENTED_METRICS,
        ),
        (
            "raw_plus_baseline_regions_augmented",
            f"Raw + baseline augmented ({region_label})",
            "hybrid",
            DEFAULT_AUGMENTED_METRICS,
        ),
    ]


def build_temporal_model_feature_frame(
    model_kind: str,
    metrics: tuple[str, ...],
    *,
    window_df: pd.DataFrame,
    raw_feature_df: pd.DataFrame,
    regions: tuple[str, ...],
) -> pd.DataFrame:
    raw_region_df = select_raw_feature_frame(raw_feature_df, regions=regions)
    if model_kind == "raw":
        return raw_region_df

    baseline_feature_df = build_feature_frame(
        window_df,
        regions=regions,
        metrics=metrics,
        prefix="baseline",
    )
    if model_kind == "baseline":
        return baseline_feature_df
    if model_kind == "hybrid":
        return merge_feature_frames([raw_region_df, baseline_feature_df])
    raise ValueError(f"Unknown temporal evidence model kind: {model_kind!r}")


def plot_temporal_evidence_summary(group_summary: pd.DataFrame, outfile: Path) -> None:
    if group_summary.empty:
        return

    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    palette = {
        "raw_regions_delta": "#1d3557",
        "baseline_regions_r2": "#6c757d",
        "baseline_regions_augmented": "#457b9d",
        "raw_plus_baseline_regions_augmented": "#e76f51",
    }

    for model_name, model_df in group_summary.groupby("model_name", observed=False):
        model_df = model_df.sort_values("target_start_ms")
        colour = palette.get(str(model_name), "#495057")
        axes[0].plot(
            model_df["target_start_ms"],
            model_df["auc_mean"],
            marker="o",
            linewidth=2.3,
            color=colour,
            label=model_df["model_label"].iloc[0],
        )
        axes[1].plot(
            model_df["target_start_ms"],
            model_df["hit_conf_rho_mean"],
            marker="o",
            linewidth=2.3,
            color=colour,
            label=model_df["model_label"].iloc[0],
        )

    axes[0].axhline(0.5, color="#6c757d", linestyle="--", linewidth=1.0)
    axes[0].set_title("Held-out hit versus miss AUC")
    axes[0].set_xlabel("Target window start (ms)")
    axes[0].set_ylabel("Subject mean AUC")
    axes[0].grid(True, alpha=0.25)

    axes[1].axhline(0.0, color="#6c757d", linestyle="--", linewidth=1.0)
    axes[1].set_title("Held-out hit confidence rho")
    axes[1].set_xlabel("Target window start (ms)")
    axes[1].set_ylabel("Subject mean Spearman rho")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_temporal_evidence_report(
    group_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Temporal Evidence Summary",
        "",
        "## Detection",
        "",
    ]

    for _, row in group_summary.sort_values(["target_start_ms", "auc_mean"], ascending=[True, False]).groupby(
        "target_start_ms",
        observed=False,
    ).head(1).iterrows():
        lines.append(
            f"- {int(row['target_start_ms'])} to {int(row['target_end_ms'])} ms: "
            f"{row['model_name']} had the strongest subject-mean AUC "
            f"({row['auc_mean']:.3f}, pooled AUC={row['pooled_auc']:.3f})."
        )

    hybrid_pairs = pairwise_summary.loc[
        (pairwise_summary["metric"] == "auc")
        & (
            (
                (pairwise_summary["model_a"] == "raw_regions_delta")
                & (pairwise_summary["model_b"] == "raw_plus_baseline_regions_augmented")
            )
            | (
                (pairwise_summary["model_b"] == "raw_regions_delta")
                & (pairwise_summary["model_a"] == "raw_plus_baseline_regions_augmented")
            )
        )
    ].copy()
    if not hybrid_pairs.empty:
        lines.extend(["", "## Hybrid gain", ""])
        for _, row in hybrid_pairs.sort_values("target_start_ms").iterrows():
            delta = float(row["mean_delta_model_b_minus_a"])
            if row["model_a"] == "raw_plus_baseline_regions_augmented":
                delta = -delta
            lines.append(
                f"- {int(row['target_start_ms'])} to {int(row['target_end_ms'])} ms: "
                f"raw plus baseline augmented minus raw delta AUC = "
                f"{delta:.3f} (p={row['ttest_p']:.3g})."
            )

    confidence_pairs = pairwise_summary.loc[
        (pairwise_summary["metric"] == "hit_conf_rho")
        & (
            (
                (pairwise_summary["model_a"] == "raw_regions_delta")
                & (pairwise_summary["model_b"] == "raw_plus_baseline_regions_augmented")
            )
            | (
                (pairwise_summary["model_b"] == "raw_regions_delta")
                & (pairwise_summary["model_a"] == "raw_plus_baseline_regions_augmented")
            )
        )
    ].copy()
    if not confidence_pairs.empty:
        lines.extend(["", "## Confidence", ""])
        for _, row in confidence_pairs.sort_values("target_start_ms").iterrows():
            delta = float(row["mean_delta_model_b_minus_a"])
            if row["model_a"] == "raw_plus_baseline_regions_augmented":
                delta = -delta
            lines.append(
                f"- {int(row['target_start_ms'])} to {int(row['target_end_ms'])} ms: "
                f"raw plus baseline augmented minus raw delta rho = "
                f"{delta:.3f} (p={row['ttest_p']:.3g})."
            )

    return "\n".join(lines)


def run_temporal_evidence_summary(
    *,
    baseline_results_dir: Path = DEFAULT_TEMPORAL_EVIDENCE_RESULTS_DIR,
    derivatives_dir: Path = DEFAULT_DERIVATIVES_DIR,
    export_dir: Path = DEFAULT_EXPORT_DIR,
    outdir: Path | None = None,
    rep_dim: int = 4,
    regions: tuple[str, ...] = DEFAULT_REGIONS,
    positive_sdt: str = DEFAULT_POSITIVE_SDT,
    negative_sdt: str = DEFAULT_NEGATIVE_SDT,
) -> int:
    outdir = baseline_results_dir / "summary_temporal_evidence" if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        baseline_df = load_baseline_region_results(baseline_results_dir)
        baseline_df = filter_temporal_baseline_rows(
            baseline_df,
            rep_dim=rep_dim,
            regions=regions,
        )
        common_trials = (
            baseline_df[["subject", "trial_idx"]]
            .drop_duplicates()
            .sort_values(["subject", "trial_idx"])
            .reset_index(drop=True)
        )
        subjects = sorted(common_trials["subject"].dropna().astype(str).unique().tolist())
        metadata_df = load_signal_metadata(
            subjects,
            derivatives_dir=derivatives_dir,
            positive_sdt=positive_sdt,
            negative_sdt=negative_sdt,
        )
        metadata_df = metadata_df.merge(
            common_trials,
            on=["subject", "trial_idx"],
            how="inner",
            validate="one_to_one",
        )
        if metadata_df.empty:
            raise ValueError("No hit or miss trials remained after aligning the temporal evidence inputs")
        metadata_df = assign_subject_twofold_splits(metadata_df)

        train_start_ms = float(sorted(baseline_df["train_start_ms"].dropna().unique().tolist())[0])
        train_end_ms = float(sorted(baseline_df["train_end_ms"].dropna().unique().tolist())[0])
        window_bounds = (
            baseline_df[["target_start_ms", "target_end_ms"]]
            .drop_duplicates()
            .sort_values(["target_start_ms", "target_end_ms"])
            .itertuples(index=False, name=None)
        )
        model_defs = build_temporal_model_definitions(regions)

        all_trial_scores = []
        all_subject_summaries = []
        all_group_summaries = []
        all_pairwise_summaries = []

        for target_start_ms, target_end_ms in window_bounds:
            window_df = baseline_df.loc[
                (baseline_df["target_start_ms"] == float(target_start_ms))
                & (baseline_df["target_end_ms"] == float(target_end_ms))
            ].copy()
            raw_feature_df = build_raw_feature_frame(
                common_trials,
                export_dir=export_dir,
                derivatives_dir=derivatives_dir,
                baseline_start_ms=train_start_ms,
                baseline_end_ms=train_end_ms,
                test_start_ms=float(target_start_ms),
                test_end_ms=float(target_end_ms),
                regions=regions,
            )

            window_trial_scores = []
            for model_name, model_label, model_kind, metrics in model_defs:
                feature_df = build_temporal_model_feature_frame(
                    model_kind,
                    metrics,
                    window_df=window_df,
                    raw_feature_df=raw_feature_df,
                    regions=regions,
                )
                trial_df = build_model_trial_table(
                    metadata_df,
                    feature_df,
                    model_name=model_name,
                    model_label=model_label,
                )
                feature_cols = [
                    column
                    for column in trial_df.columns
                    if column.startswith(("raw_", "baseline_"))
                ]
                scores = fit_out_of_fold_evidence(
                    trial_df,
                    feature_cols=feature_cols,
                )
                scores["target_start_ms"] = float(target_start_ms)
                scores["target_end_ms"] = float(target_end_ms)
                scores["target_center_ms"] = 0.5 * (
                    float(target_start_ms) + float(target_end_ms)
                )
                window_trial_scores.append(scores)

            trial_scores = pd.concat(window_trial_scores, ignore_index=True)
            subject_summary = evaluate_subject_metrics(
                trial_scores,
                positive_sdt=positive_sdt,
            )
            subject_summary["target_start_ms"] = float(target_start_ms)
            subject_summary["target_end_ms"] = float(target_end_ms)
            subject_summary["target_center_ms"] = 0.5 * (
                float(target_start_ms) + float(target_end_ms)
            )
            group_summary = summarise_model_metrics(
                trial_scores,
                subject_summary,
                positive_sdt=positive_sdt,
            )
            group_summary["target_start_ms"] = float(target_start_ms)
            group_summary["target_end_ms"] = float(target_end_ms)
            group_summary["target_center_ms"] = 0.5 * (
                float(target_start_ms) + float(target_end_ms)
            )
            pairwise_summary = build_pairwise_model_comparisons(subject_summary)
            pairwise_summary["target_start_ms"] = float(target_start_ms)
            pairwise_summary["target_end_ms"] = float(target_end_ms)
            pairwise_summary["target_center_ms"] = 0.5 * (
                float(target_start_ms) + float(target_end_ms)
            )

            all_trial_scores.append(trial_scores)
            all_subject_summaries.append(subject_summary)
            all_group_summaries.append(group_summary)
            all_pairwise_summaries.append(pairwise_summary)

        trial_scores_df = pd.concat(all_trial_scores, ignore_index=True)
        subject_summary_df = pd.concat(all_subject_summaries, ignore_index=True)
        group_summary_df = pd.concat(all_group_summaries, ignore_index=True)
        pairwise_summary_df = pd.concat(all_pairwise_summaries, ignore_index=True)
        report = build_temporal_evidence_report(group_summary_df, pairwise_summary_df)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    trial_scores_path = outdir / "temporal_evidence_trial_scores.csv"
    subject_summary_path = outdir / "temporal_evidence_subject_summary.csv"
    group_summary_path = outdir / "temporal_evidence_group_summary.csv"
    pairwise_summary_path = outdir / "temporal_evidence_pairwise_summary.csv"
    plot_path = outdir / "temporal_evidence_timecourse.png"
    report_path = outdir / "temporal_evidence_summary.md"

    save_table(trial_scores_df, trial_scores_path)
    save_table(subject_summary_df, subject_summary_path)
    save_table(group_summary_df, group_summary_path)
    save_table(pairwise_summary_df, pairwise_summary_path)
    plot_temporal_evidence_summary(group_summary_df, plot_path)
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"Saved {trial_scores_path}")
    print(f"Saved {subject_summary_path}")
    print(f"Saved {group_summary_path}")
    print(f"Saved {pairwise_summary_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {report_path}")
    print("\nGroup summary:")
    print(
        group_summary_df[
            [
                "target_start_ms",
                "target_end_ms",
                "model_name",
                "auc_mean",
                "auc_std",
                "auc_ttest_p",
                "pooled_auc",
                "hit_conf_rho_mean",
                "pooled_hit_conf_rho",
            ]
        ].to_string(index=False)
    )
    return 0
