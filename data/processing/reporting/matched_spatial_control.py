"""Amplitude-matched spatial-control evidence summary for one focus region."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from eegprep import DEFAULT_DERIVATIVES_DIR, load_exported_subject, load_subject_channel_labels
from experiments.spatial import build_channel_groups, build_size_matched_control_groups
from experiments.temporal import infer_boundary_tolerance_ms, slice_trial_window
from framework.summaries import prepare_pyplot, save_table
from reporting.evidence import (
    DEFAULT_EXPORT_DIR,
    DEFAULT_NEGATIVE_SDT,
    DEFAULT_POSITIVE_SDT,
    assign_subject_twofold_splits,
    build_feature_frame,
    build_model_trial_table,
    fit_out_of_fold_evidence,
    load_baseline_region_results,
    load_signal_metadata,
    merge_feature_frames,
)
from reporting.matched_stimamp_evidence import (
    build_match_quality_summary,
    build_matched_pairs,
    compare_pair_scores,
    merge_stimamp,
    safe_wilcoxon,
)
from reporting.statistics import (
    DEFAULT_N_BOOT,
    DEFAULT_RANDOM_SEED,
    bootstrap_mean_ci,
)


DEFAULT_BASELINE_RESULTS_DIR = Path(
    "./data/results_baseline/region_sliding_baseline300ms_controls_focus_q4"
)
DEFAULT_AUGMENTED_METRICS = (
    "pred_r2_obs",
    "pred_mse_obs",
    "pred_r2_latent",
    "pred_nll_latent",
)
DEFAULT_MODEL_DEFS = (
    ("raw_regions_delta", "Raw regional delta", "raw", ()),
    (
        "baseline_regions_augmented",
        "Baseline regional augmented",
        "baseline",
        DEFAULT_AUGMENTED_METRICS,
    ),
    (
        "raw_plus_baseline_regions_augmented",
        "Raw + baseline regional augmented",
        "hybrid",
        DEFAULT_AUGMENTED_METRICS,
    ),
)
DEFAULT_RAW_MODEL = "raw_regions_delta"
DEFAULT_HYBRID_MODEL = "raw_plus_baseline_regions_augmented"
DEFAULT_FOCUS_REGION = "central"
DEFAULT_TARGET_START_MS = 125.0
DEFAULT_TARGET_END_MS = 375.0
DEFAULT_REP_DIM = 4


def filter_focus_region_rows(
    baseline_df: pd.DataFrame,
    *,
    rep_dim: int,
    focus_region: str,
    target_start_ms: float,
    target_end_ms: float,
) -> pd.DataFrame:
    """Keep one named region and its size-matched controls at one window."""
    filtered = baseline_df.loc[
        (baseline_df["rep_dim"] == int(rep_dim))
        & (baseline_df["matched_region_name"] == str(focus_region))
        & (baseline_df["target_start_ms"] == float(target_start_ms))
        & (baseline_df["target_end_ms"] == float(target_end_ms))
        & (baseline_df["group_kind"].isin(["named_region", "size_matched_control"]))
    ].copy()
    if filtered.empty:
        raise ValueError("No baseline rows matched the requested spatial-control setup")

    present_named = filtered.loc[filtered["group_kind"] == "named_region", "region_name"].unique().tolist()
    if present_named != [focus_region]:
        raise ValueError(
            f"Expected one named-region row for {focus_region!r}, found {present_named}"
        )
    return filtered.sort_values(
        ["subject", "trial_idx", "group_kind", "region_name"]
    ).reset_index(drop=True)


def collect_complete_trials(focus_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only trials that are available for the named region and every control draw."""
    required_groups = (
        focus_df[["group_kind", "region_name"]]
        .drop_duplicates()
        .shape[0]
    )
    counts = (
        focus_df.groupby(["subject", "trial_idx"], observed=False)["region_name"]
        .nunique()
        .rename("n_groups")
        .reset_index()
    )
    common_trials = counts.loc[counts["n_groups"] == required_groups, ["subject", "trial_idx"]]
    if common_trials.empty:
        raise ValueError("No subject-trial pairs were available for every selected group")
    return common_trials.sort_values(["subject", "trial_idx"]).reset_index(drop=True)


def infer_reference_labels(subject_labels: list[str], reference_labels: list[str] | None) -> list[str]:
    """Check that channel labels are compatible across subjects."""
    cleaned = [str(label).strip() for label in subject_labels]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("Channel labels must be unique within each subject")
    if reference_labels is None:
        return cleaned

    if set(cleaned) != set(reference_labels):
        missing = sorted(set(reference_labels).difference(cleaned))
        extra = sorted(set(cleaned).difference(reference_labels))
        raise ValueError(
            "Channel labels differed across subjects. "
            f"Missing labels: {missing[:5]}; extra labels: {extra[:5]}"
        )
    return reference_labels


def build_subject_group_index_map(
    subjects: list[str],
    *,
    focus_region: str,
    n_control_draws: int,
    random_state: int,
    derivatives_dir: Path,
) -> tuple[list[str], dict[str, list[str]], dict[str, dict[str, np.ndarray]]]:
    """Rebuild the named region and the deterministic size-matched controls."""
    reference_labels: list[str] | None = None
    subject_labels: dict[str, list[str]] = {}
    subject_groups: dict[str, dict[str, np.ndarray]] = {}

    for subject in subjects:
        labels = load_subject_channel_labels(subject, derivatives_dir=derivatives_dir)
        reference_labels = infer_reference_labels(labels, reference_labels)
        labels = [str(label).strip() for label in labels]

        named_groups = build_channel_groups(labels)
        if focus_region not in named_groups:
            raise ValueError(f"{subject} does not contain the requested region {focus_region!r}")

        control_groups = build_size_matched_control_groups(
            labels,
            {focus_region: np.asarray(named_groups[focus_region], dtype=int)},
            n_draws=int(n_control_draws),
            random_state=int(random_state + sum(ord(char) for char in subject)),
        )

        groups = {focus_region: np.asarray(named_groups[focus_region], dtype=int)}
        for draw_idx in range(1, int(n_control_draws) + 1):
            control_name = f"{focus_region}_control_{draw_idx:02d}"
            groups[control_name] = np.asarray(control_groups[(focus_region, draw_idx)], dtype=int)

        subject_labels[subject] = labels
        subject_groups[subject] = groups

    if reference_labels is None:
        raise ValueError("No channel labels were available")
    return reference_labels, subject_labels, subject_groups


def build_group_raw_feature_frame(
    common_trials: pd.DataFrame,
    *,
    export_dir: Path,
    subject_labels: dict[str, list[str]],
    subject_groups: dict[str, dict[str, np.ndarray]],
    reference_labels: list[str],
    baseline_start_ms: float,
    baseline_end_ms: float,
    target_start_ms: float,
    target_end_ms: float,
) -> pd.DataFrame:
    """Build masked raw delta features for the named region and its controls."""
    rows: list[dict[str, object]] = []
    reference_index = {label: idx for idx, label in enumerate(reference_labels)}
    feature_columns = [f"raw_chan__{label}" for label in reference_labels]

    for subject, subject_trials in common_trials.groupby("subject", observed=False):
        subject_path = export_dir / f"{subject}_preproc_01hz_export.mat"
        data, _sfreq, times_ms = load_exported_subject(subject_path)
        labels = subject_labels[str(subject)]
        label_index = {label: idx for idx, label in enumerate(labels)}
        if data.shape[2] != len(labels):
            raise ValueError(
                f"Channel count mismatch for {subject}: data has {data.shape[2]} channels "
                f"but metadata has {len(labels)} labels"
            )

        boundary_tolerance_ms = infer_boundary_tolerance_ms(times_ms)
        group_map = subject_groups[str(subject)]
        for trial_idx in subject_trials["trial_idx"].astype(int).tolist():
            if trial_idx < 0 or trial_idx >= data.shape[0]:
                raise IndexError(
                    f"Trial index {trial_idx} is out of range for {subject} with {data.shape[0]} trials"
                )

            X = np.asarray(data[trial_idx], dtype=float)
            baseline_X, _ = slice_trial_window(
                X,
                times_ms,
                baseline_start_ms,
                baseline_end_ms,
                require_full_window=True,
                boundary_tolerance_ms=boundary_tolerance_ms,
            )
            target_X, _ = slice_trial_window(
                X,
                times_ms,
                target_start_ms,
                target_end_ms,
                require_full_window=True,
                boundary_tolerance_ms=boundary_tolerance_ms,
            )
            delta = target_X.mean(axis=0) - baseline_X.mean(axis=0)

            for region_name, region_idx in group_map.items():
                masked_delta = np.zeros(len(reference_labels), dtype=float)
                for chan_idx in np.asarray(region_idx, dtype=int):
                    label = labels[int(chan_idx)]
                    ref_idx = reference_index.get(label)
                    if ref_idx is None:
                        ref_idx = reference_index[reference_labels[label_index[label]]]
                    masked_delta[int(ref_idx)] = float(delta[int(chan_idx)])

                row = {
                    "subject": str(subject),
                    "trial_idx": int(trial_idx),
                    "region_name": str(region_name),
                }
                row.update(
                    {
                        feature_name: float(masked_delta[col_idx])
                        for col_idx, feature_name in enumerate(feature_columns)
                    }
                )
                rows.append(row)

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError("No raw feature rows could be built for the spatial-control summary")
    return raw_df.sort_values(["subject", "trial_idx", "region_name"]).reset_index(drop=True)


def select_group_raw_feature_frame(raw_df: pd.DataFrame, *, region_name: str) -> pd.DataFrame:
    """Keep the raw EEG features for one named region or one control draw."""
    filtered = raw_df.loc[raw_df["region_name"] == str(region_name)].copy()
    if filtered.empty:
        raise ValueError(f"No raw EEG rows were available for {region_name!r}")

    feature_cols = [column for column in filtered.columns if column.startswith("raw_")]
    return filtered.loc[:, ["subject", "trial_idx", *feature_cols]].reset_index(drop=True)


def build_group_model_feature_frame(
    model_kind: str,
    metrics: tuple[str, ...],
    *,
    group_baseline_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    region_name: str,
) -> pd.DataFrame:
    """Build one feature table for one region/control group."""
    raw_group_df = select_group_raw_feature_frame(raw_df, region_name=region_name)
    if model_kind == "raw":
        return raw_group_df

    baseline_feature_df = build_feature_frame(
        group_baseline_df,
        regions=(str(region_name),),
        metrics=metrics,
        prefix="baseline",
    )
    if model_kind == "baseline":
        return baseline_feature_df
    if model_kind == "hybrid":
        return merge_feature_frames([raw_group_df, baseline_feature_df])
    raise ValueError(f"Unknown spatial-control model kind: {model_kind!r}")


def evaluate_subject_pair_metrics_by_group(
    trial_scores: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise matched-pair performance per subject, model, and channel group."""
    rows = []
    merge_columns = ["target_start_ms", "target_end_ms", "target_center_ms", "subject"]
    group_columns = ["group_kind", "matched_region_name", "region_name", "model_name", "model_label"]

    for group_values, group_df in trial_scores.groupby(group_columns, observed=False):
        group_kind, matched_region_name, region_name, model_name, model_label = group_values
        positive_scores = pair_df.merge(
            group_df[merge_columns + ["trial_idx", "evidence_score"]],
            left_on=merge_columns + ["positive_trial_idx"],
            right_on=merge_columns + ["trial_idx"],
            how="left",
            validate="many_to_one",
        ).rename(columns={"evidence_score": "positive_score"})
        negative_scores = pair_df.merge(
            group_df[merge_columns + ["trial_idx", "evidence_score"]],
            left_on=merge_columns + ["negative_trial_idx"],
            right_on=merge_columns + ["trial_idx"],
            how="left",
            validate="many_to_one",
        ).rename(columns={"evidence_score": "negative_score"})

        merged = positive_scores[
            merge_columns + ["pair_idx", "stimamp_abs_diff", "positive_score"]
        ].merge(
            negative_scores[merge_columns + ["pair_idx", "negative_score"]],
            on=merge_columns + ["pair_idx"],
            how="inner",
            validate="one_to_one",
        )
        merged = merged.dropna(subset=["positive_score", "negative_score"]).copy()
        merged["pair_success"] = compare_pair_scores(
            merged["positive_score"].to_numpy(dtype=float),
            merged["negative_score"].to_numpy(dtype=float),
        )
        merged["score_margin"] = (
            merged["positive_score"].to_numpy(dtype=float)
            - merged["negative_score"].to_numpy(dtype=float)
        )

        control_draw_idx = np.nan
        if str(group_kind) == "size_matched_control":
            suffix = str(region_name).rsplit("_", 1)[-1]
            try:
                control_draw_idx = int(suffix)
            except ValueError:
                control_draw_idx = np.nan

        for keys, subject_df in merged.groupby(merge_columns, observed=False):
            target_start_ms, target_end_ms, target_center_ms, subject = keys
            rows.append(
                {
                    "target_start_ms": float(target_start_ms),
                    "target_end_ms": float(target_end_ms),
                    "target_center_ms": float(target_center_ms),
                    "subject": str(subject),
                    "group_kind": str(group_kind),
                    "matched_region_name": str(matched_region_name),
                    "region_name": str(region_name),
                    "control_draw_idx": control_draw_idx,
                    "model_name": str(model_name),
                    "model_label": str(model_label),
                    "n_pairs": int(subject_df.shape[0]),
                    "pair_acc": float(subject_df["pair_success"].mean()),
                    "mean_margin": float(subject_df["score_margin"].mean()),
                    "mean_stimamp_abs_diff": float(subject_df["stimamp_abs_diff"].mean()),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No subject-level matched-pair summaries could be built")
    return out.sort_values(
        ["region_name", "model_name", "subject"]
    ).reset_index(drop=True)


def summarise_group_pair_metrics(
    subject_summary: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    """Aggregate matched-pair metrics across subjects for each group and model."""
    rows = []
    group_columns = [
        "target_start_ms",
        "target_end_ms",
        "target_center_ms",
        "group_kind",
        "matched_region_name",
        "region_name",
        "model_name",
        "model_label",
    ]
    for group_idx, (group_values, group_df) in enumerate(
        subject_summary.groupby(group_columns, observed=False)
    ):
        (
            target_start_ms,
            target_end_ms,
            target_center_ms,
            group_kind,
            matched_region_name,
            region_name,
            model_name,
            model_label,
        ) = group_values
        pair_acc_values = group_df["pair_acc"].dropna().to_numpy(dtype=float)
        margin_values = group_df["mean_margin"].dropna().to_numpy(dtype=float)
        pair_acc_t = (
            stats.ttest_1samp(pair_acc_values, 0.5, nan_policy="omit")
            if pair_acc_values.size >= 2
            else None
        )
        margin_t = (
            stats.ttest_1samp(margin_values, 0.0, nan_policy="omit")
            if margin_values.size >= 2
            else None
        )
        pair_acc_ci_low, pair_acc_ci_high = bootstrap_mean_ci(
            pair_acc_values,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 1_000 + group_idx,
        )
        margin_ci_low, margin_ci_high = bootstrap_mean_ci(
            margin_values,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 20_000 + group_idx,
        )
        pair_acc_w_stat, pair_acc_w_p = safe_wilcoxon(pair_acc_values, null_value=0.5)
        margin_w_stat, margin_w_p = safe_wilcoxon(margin_values, null_value=0.0)

        rows.append(
            {
                "target_start_ms": float(target_start_ms),
                "target_end_ms": float(target_end_ms),
                "target_center_ms": float(target_center_ms),
                "group_kind": str(group_kind),
                "matched_region_name": str(matched_region_name),
                "region_name": str(region_name),
                "control_draw_idx": group_df["control_draw_idx"].iloc[0],
                "model_name": str(model_name),
                "model_label": str(model_label),
                "n_subjects": int(group_df.shape[0]),
                "pair_acc_mean": float(group_df["pair_acc"].mean()),
                "pair_acc_std": float(group_df["pair_acc"].std(ddof=1))
                if group_df.shape[0] > 1
                else float("nan"),
                "pair_acc_ci95_low": pair_acc_ci_low,
                "pair_acc_ci95_high": pair_acc_ci_high,
                "pair_acc_above_half_share": float(np.mean(group_df["pair_acc"] > 0.5)),
                "pair_acc_ttest_p": float(pair_acc_t.pvalue) if pair_acc_t is not None else float("nan"),
                "pair_acc_wilcoxon_stat": pair_acc_w_stat,
                "pair_acc_wilcoxon_p": pair_acc_w_p,
                "mean_margin_mean": float(group_df["mean_margin"].mean()),
                "mean_margin_std": float(group_df["mean_margin"].std(ddof=1))
                if group_df.shape[0] > 1
                else float("nan"),
                "mean_margin_ci95_low": margin_ci_low,
                "mean_margin_ci95_high": margin_ci_high,
                "mean_margin_positive_share": float(np.mean(group_df["mean_margin"] > 0.0)),
                "mean_margin_ttest_p": float(margin_t.pvalue) if margin_t is not None else float("nan"),
                "mean_margin_wilcoxon_stat": margin_w_stat,
                "mean_margin_wilcoxon_p": margin_w_p,
                "mean_pairs_per_subject": float(group_df["n_pairs"].mean()),
                "mean_stimamp_abs_diff": float(group_df["mean_stimamp_abs_diff"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["region_name", "model_name"]).reset_index(drop=True)


def build_hybrid_gain_subject_summary(
    subject_summary: pd.DataFrame,
    *,
    raw_model: str,
    hybrid_model: str,
) -> pd.DataFrame:
    """Compute the per-subject hybrid-over-raw gain for each group."""
    merge_columns = [
        "target_start_ms",
        "target_end_ms",
        "target_center_ms",
        "subject",
        "group_kind",
        "matched_region_name",
        "region_name",
    ]
    raw_df = subject_summary.loc[subject_summary["model_name"] == str(raw_model)].copy()
    hybrid_df = subject_summary.loc[subject_summary["model_name"] == str(hybrid_model)].copy()
    if raw_df.empty or hybrid_df.empty:
        raise ValueError("Both raw and hybrid matched-pair summaries are required")

    raw_df = raw_df.rename(
        columns={
            "control_draw_idx": "raw_control_draw_idx",
            "pair_acc": "raw_pair_acc",
            "mean_margin": "raw_mean_margin",
            "n_pairs": "raw_n_pairs",
        }
    )
    hybrid_df = hybrid_df.rename(
        columns={
            "control_draw_idx": "hybrid_control_draw_idx",
            "pair_acc": "hybrid_pair_acc",
            "mean_margin": "hybrid_mean_margin",
            "n_pairs": "hybrid_n_pairs",
        }
    )
    merged = raw_df[merge_columns + ["raw_control_draw_idx", "raw_pair_acc", "raw_mean_margin", "raw_n_pairs"]].merge(
        hybrid_df[
            merge_columns
            + ["hybrid_control_draw_idx", "hybrid_pair_acc", "hybrid_mean_margin", "hybrid_n_pairs"]
        ],
        on=merge_columns,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping raw and hybrid subject rows were available")

    merged["control_draw_idx"] = merged["raw_control_draw_idx"].combine_first(
        merged["hybrid_control_draw_idx"]
    )
    merged["pair_acc_gain"] = merged["hybrid_pair_acc"] - merged["raw_pair_acc"]
    merged["mean_margin_gain"] = merged["hybrid_mean_margin"] - merged["raw_mean_margin"]
    merged["n_pairs"] = merged[["raw_n_pairs", "hybrid_n_pairs"]].min(axis=1)
    return merged.sort_values(["region_name", "subject"]).reset_index(drop=True)


def summarise_hybrid_gain(
    gain_subject_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    """Aggregate the hybrid-over-raw gain across subjects for each group."""
    rows = []
    group_columns = [
        "target_start_ms",
        "target_end_ms",
        "target_center_ms",
        "group_kind",
        "matched_region_name",
        "region_name",
    ]
    for group_idx, (group_values, group_df) in enumerate(
        gain_subject_df.groupby(group_columns, observed=False)
    ):
        (
            target_start_ms,
            target_end_ms,
            target_center_ms,
            group_kind,
            matched_region_name,
            region_name,
        ) = group_values
        pair_acc_values = group_df["pair_acc_gain"].dropna().to_numpy(dtype=float)
        margin_values = group_df["mean_margin_gain"].dropna().to_numpy(dtype=float)
        pair_acc_t = (
            stats.ttest_1samp(pair_acc_values, 0.0, nan_policy="omit")
            if pair_acc_values.size >= 2
            else None
        )
        margin_t = (
            stats.ttest_1samp(margin_values, 0.0, nan_policy="omit")
            if margin_values.size >= 2
            else None
        )
        pair_acc_ci_low, pair_acc_ci_high = bootstrap_mean_ci(
            pair_acc_values,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 40_000 + group_idx,
        )
        margin_ci_low, margin_ci_high = bootstrap_mean_ci(
            margin_values,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 60_000 + group_idx,
        )
        pair_acc_w_stat, pair_acc_w_p = safe_wilcoxon(pair_acc_values, null_value=0.0)
        margin_w_stat, margin_w_p = safe_wilcoxon(margin_values, null_value=0.0)

        rows.append(
            {
                "target_start_ms": float(target_start_ms),
                "target_end_ms": float(target_end_ms),
                "target_center_ms": float(target_center_ms),
                "group_kind": str(group_kind),
                "matched_region_name": str(matched_region_name),
                "region_name": str(region_name),
                "control_draw_idx": group_df["control_draw_idx"].iloc[0],
                "n_subjects": int(group_df.shape[0]),
                "pair_acc_gain_mean": float(group_df["pair_acc_gain"].mean()),
                "pair_acc_gain_std": float(group_df["pair_acc_gain"].std(ddof=1))
                if group_df.shape[0] > 1
                else float("nan"),
                "pair_acc_gain_ci95_low": pair_acc_ci_low,
                "pair_acc_gain_ci95_high": pair_acc_ci_high,
                "pair_acc_gain_positive_share": float(np.mean(group_df["pair_acc_gain"] > 0.0)),
                "pair_acc_gain_ttest_p": float(pair_acc_t.pvalue) if pair_acc_t is not None else float("nan"),
                "pair_acc_gain_wilcoxon_stat": pair_acc_w_stat,
                "pair_acc_gain_wilcoxon_p": pair_acc_w_p,
                "mean_margin_gain_mean": float(group_df["mean_margin_gain"].mean()),
                "mean_margin_gain_std": float(group_df["mean_margin_gain"].std(ddof=1))
                if group_df.shape[0] > 1
                else float("nan"),
                "mean_margin_gain_ci95_low": margin_ci_low,
                "mean_margin_gain_ci95_high": margin_ci_high,
                "mean_margin_gain_positive_share": float(np.mean(group_df["mean_margin_gain"] > 0.0)),
                "mean_margin_gain_ttest_p": float(margin_t.pvalue) if margin_t is not None else float("nan"),
                "mean_margin_gain_wilcoxon_stat": margin_w_stat,
                "mean_margin_gain_wilcoxon_p": margin_w_p,
                "mean_pairs_per_subject": float(group_df["n_pairs"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["group_kind", "region_name"]).reset_index(drop=True)


def build_named_vs_control_subject_summary(gain_subject_df: pd.DataFrame) -> pd.DataFrame:
    """Compare the named-region gain against the mean control gain within subject."""
    named_df = gain_subject_df.loc[gain_subject_df["group_kind"] == "named_region"].copy()
    control_df = gain_subject_df.loc[gain_subject_df["group_kind"] == "size_matched_control"].copy()
    if named_df.empty or control_df.empty:
        return pd.DataFrame()

    merge_keys = [
        "subject",
        "matched_region_name",
        "target_start_ms",
        "target_end_ms",
        "target_center_ms",
    ]
    control_means = (
        control_df.groupby(merge_keys, observed=False)
        .agg(
            control_mean_pair_acc_gain=("pair_acc_gain", "mean"),
            control_mean_margin_gain=("mean_margin_gain", "mean"),
            n_control_draws=("region_name", "nunique"),
        )
        .reset_index()
    )
    named_df = named_df.rename(
        columns={
            "region_name": "named_region_name",
            "pair_acc_gain": "named_pair_acc_gain",
            "mean_margin_gain": "named_mean_margin_gain",
        }
    )[
        merge_keys + ["named_region_name", "named_pair_acc_gain", "named_mean_margin_gain"]
    ]
    merged = named_df.merge(
        control_means,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["pair_acc_gain_minus_control"] = (
        merged["named_pair_acc_gain"] - merged["control_mean_pair_acc_gain"]
    )
    merged["mean_margin_gain_minus_control"] = (
        merged["named_mean_margin_gain"] - merged["control_mean_margin_gain"]
    )
    return merged.sort_values(["matched_region_name", "subject"]).reset_index(drop=True)


def build_named_vs_control_model_subject_summary(
    subject_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare named-region absolute matched-pair performance against the mean control performance."""
    named_df = subject_summary.loc[subject_summary["group_kind"] == "named_region"].copy()
    control_df = subject_summary.loc[subject_summary["group_kind"] == "size_matched_control"].copy()
    if named_df.empty or control_df.empty:
        return pd.DataFrame()

    merge_keys = [
        "subject",
        "matched_region_name",
        "target_start_ms",
        "target_end_ms",
        "target_center_ms",
        "model_name",
        "model_label",
    ]
    control_means = (
        control_df.groupby(merge_keys, observed=False)
        .agg(
            control_mean_pair_acc=("pair_acc", "mean"),
            control_mean_margin=("mean_margin", "mean"),
            n_control_draws=("region_name", "nunique"),
        )
        .reset_index()
    )
    named_df = named_df.rename(
        columns={
            "region_name": "named_region_name",
            "pair_acc": "named_pair_acc",
            "mean_margin": "named_mean_margin",
        }
    )[
        merge_keys + ["named_region_name", "named_pair_acc", "named_mean_margin"]
    ]
    merged = named_df.merge(
        control_means,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["pair_acc_minus_control"] = (
        merged["named_pair_acc"] - merged["control_mean_pair_acc"]
    )
    merged["mean_margin_minus_control"] = (
        merged["named_mean_margin"] - merged["control_mean_margin"]
    )
    return merged.sort_values(["model_name", "subject"]).reset_index(drop=True)


def summarise_named_vs_control(
    comparison_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    """Aggregate the named-region advantage over the size-matched controls."""
    rows = []
    group_columns = [
        "matched_region_name",
        "target_start_ms",
        "target_end_ms",
        "target_center_ms",
    ]
    for group_idx, (group_values, group_df) in enumerate(
        comparison_df.groupby(group_columns, observed=False)
    ):
        matched_region_name, target_start_ms, target_end_ms, target_center_ms = group_values
        pair_acc_values = group_df["pair_acc_gain_minus_control"].dropna().to_numpy(dtype=float)
        margin_values = group_df["mean_margin_gain_minus_control"].dropna().to_numpy(dtype=float)
        pair_acc_t = (
            stats.ttest_1samp(pair_acc_values, 0.0, nan_policy="omit")
            if pair_acc_values.size >= 2
            else None
        )
        margin_t = (
            stats.ttest_1samp(margin_values, 0.0, nan_policy="omit")
            if margin_values.size >= 2
            else None
        )
        pair_acc_ci_low, pair_acc_ci_high = bootstrap_mean_ci(
            pair_acc_values,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 80_000 + group_idx,
        )
        margin_ci_low, margin_ci_high = bootstrap_mean_ci(
            margin_values,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 100_000 + group_idx,
        )
        pair_acc_w_stat, pair_acc_w_p = safe_wilcoxon(pair_acc_values, null_value=0.0)
        margin_w_stat, margin_w_p = safe_wilcoxon(margin_values, null_value=0.0)

        rows.append(
            {
                "matched_region_name": str(matched_region_name),
                "target_start_ms": float(target_start_ms),
                "target_end_ms": float(target_end_ms),
                "target_center_ms": float(target_center_ms),
                "n_subjects": int(group_df.shape[0]),
                "n_control_draws": int(group_df["n_control_draws"].iloc[0]),
                "named_pair_acc_gain_mean": float(group_df["named_pair_acc_gain"].mean()),
                "control_pair_acc_gain_mean": float(group_df["control_mean_pair_acc_gain"].mean()),
                "pair_acc_gain_minus_control_mean": float(np.mean(pair_acc_values)),
                "pair_acc_gain_minus_control_std": float(np.std(pair_acc_values, ddof=1))
                if pair_acc_values.size > 1
                else float("nan"),
                "pair_acc_gain_minus_control_ci95_low": pair_acc_ci_low,
                "pair_acc_gain_minus_control_ci95_high": pair_acc_ci_high,
                "pair_acc_gain_minus_control_positive_share": float(np.mean(pair_acc_values > 0.0)),
                "pair_acc_gain_minus_control_ttest_p": float(pair_acc_t.pvalue)
                if pair_acc_t is not None
                else float("nan"),
                "pair_acc_gain_minus_control_wilcoxon_stat": pair_acc_w_stat,
                "pair_acc_gain_minus_control_wilcoxon_p": pair_acc_w_p,
                "named_mean_margin_gain_mean": float(group_df["named_mean_margin_gain"].mean()),
                "control_mean_margin_gain_mean": float(group_df["control_mean_margin_gain"].mean()),
                "mean_margin_gain_minus_control_mean": float(np.mean(margin_values)),
                "mean_margin_gain_minus_control_std": float(np.std(margin_values, ddof=1))
                if margin_values.size > 1
                else float("nan"),
                "mean_margin_gain_minus_control_ci95_low": margin_ci_low,
                "mean_margin_gain_minus_control_ci95_high": margin_ci_high,
                "mean_margin_gain_minus_control_positive_share": float(np.mean(margin_values > 0.0)),
                "mean_margin_gain_minus_control_ttest_p": float(margin_t.pvalue)
                if margin_t is not None
                else float("nan"),
                "mean_margin_gain_minus_control_wilcoxon_stat": margin_w_stat,
                "mean_margin_gain_minus_control_wilcoxon_p": margin_w_p,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["matched_region_name", "target_start_ms"]).reset_index(drop=True)


def summarise_named_vs_control_model(
    comparison_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    """Aggregate named-versus-control absolute matched-pair performance for each model."""
    rows = []
    group_columns = [
        "matched_region_name",
        "target_start_ms",
        "target_end_ms",
        "target_center_ms",
        "model_name",
        "model_label",
    ]
    for group_idx, (group_values, group_df) in enumerate(
        comparison_df.groupby(group_columns, observed=False)
    ):
        (
            matched_region_name,
            target_start_ms,
            target_end_ms,
            target_center_ms,
            model_name,
            model_label,
        ) = group_values
        pair_acc_values = group_df["pair_acc_minus_control"].dropna().to_numpy(dtype=float)
        margin_values = group_df["mean_margin_minus_control"].dropna().to_numpy(dtype=float)
        pair_acc_t = (
            stats.ttest_1samp(pair_acc_values, 0.0, nan_policy="omit")
            if pair_acc_values.size >= 2
            else None
        )
        margin_t = (
            stats.ttest_1samp(margin_values, 0.0, nan_policy="omit")
            if margin_values.size >= 2
            else None
        )
        pair_acc_ci_low, pair_acc_ci_high = bootstrap_mean_ci(
            pair_acc_values,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 120_000 + group_idx,
        )
        margin_ci_low, margin_ci_high = bootstrap_mean_ci(
            margin_values,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 140_000 + group_idx,
        )
        pair_acc_w_stat, pair_acc_w_p = safe_wilcoxon(pair_acc_values, null_value=0.0)
        margin_w_stat, margin_w_p = safe_wilcoxon(margin_values, null_value=0.0)

        rows.append(
            {
                "matched_region_name": str(matched_region_name),
                "target_start_ms": float(target_start_ms),
                "target_end_ms": float(target_end_ms),
                "target_center_ms": float(target_center_ms),
                "model_name": str(model_name),
                "model_label": str(model_label),
                "n_subjects": int(group_df.shape[0]),
                "n_control_draws": int(group_df["n_control_draws"].iloc[0]),
                "named_pair_acc_mean": float(group_df["named_pair_acc"].mean()),
                "control_pair_acc_mean": float(group_df["control_mean_pair_acc"].mean()),
                "pair_acc_minus_control_mean": float(np.mean(pair_acc_values)),
                "pair_acc_minus_control_std": float(np.std(pair_acc_values, ddof=1))
                if pair_acc_values.size > 1
                else float("nan"),
                "pair_acc_minus_control_ci95_low": pair_acc_ci_low,
                "pair_acc_minus_control_ci95_high": pair_acc_ci_high,
                "pair_acc_minus_control_positive_share": float(np.mean(pair_acc_values > 0.0)),
                "pair_acc_minus_control_ttest_p": float(pair_acc_t.pvalue)
                if pair_acc_t is not None
                else float("nan"),
                "pair_acc_minus_control_wilcoxon_stat": pair_acc_w_stat,
                "pair_acc_minus_control_wilcoxon_p": pair_acc_w_p,
                "named_mean_margin_mean": float(group_df["named_mean_margin"].mean()),
                "control_mean_margin_mean": float(group_df["control_mean_margin"].mean()),
                "mean_margin_minus_control_mean": float(np.mean(margin_values)),
                "mean_margin_minus_control_std": float(np.std(margin_values, ddof=1))
                if margin_values.size > 1
                else float("nan"),
                "mean_margin_minus_control_ci95_low": margin_ci_low,
                "mean_margin_minus_control_ci95_high": margin_ci_high,
                "mean_margin_minus_control_positive_share": float(np.mean(margin_values > 0.0)),
                "mean_margin_minus_control_ttest_p": float(margin_t.pvalue)
                if margin_t is not None
                else float("nan"),
                "mean_margin_minus_control_wilcoxon_stat": margin_w_stat,
                "mean_margin_minus_control_wilcoxon_p": margin_w_p,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["target_start_ms", "model_name"]).reset_index(drop=True)


def build_control_draw_rank_summary(gain_group_summary: pd.DataFrame) -> pd.DataFrame:
    """Place the named-region gain against the across-subject control draws."""
    rows = []
    window_columns = [
        "matched_region_name",
        "target_start_ms",
        "target_end_ms",
        "target_center_ms",
    ]
    for window_values, window_df in gain_group_summary.groupby(window_columns, observed=False):
        matched_region_name, target_start_ms, target_end_ms, target_center_ms = window_values
        named_df = window_df.loc[window_df["group_kind"] == "named_region"].copy()
        control_df = window_df.loc[window_df["group_kind"] == "size_matched_control"].copy()
        if named_df.empty or control_df.empty:
            continue

        named_pair_acc = float(named_df["pair_acc_gain_mean"].iloc[0])
        named_margin = float(named_df["mean_margin_gain_mean"].iloc[0])
        control_pair_acc = control_df["pair_acc_gain_mean"].to_numpy(dtype=float)
        control_margin = control_df["mean_margin_gain_mean"].to_numpy(dtype=float)
        rows.append(
            {
                "matched_region_name": str(matched_region_name),
                "target_start_ms": float(target_start_ms),
                "target_end_ms": float(target_end_ms),
                "target_center_ms": float(target_center_ms),
                "n_control_draws": int(control_df.shape[0]),
                "named_pair_acc_gain_mean": named_pair_acc,
                "control_pair_acc_gain_min": float(np.min(control_pair_acc)),
                "control_pair_acc_gain_max": float(np.max(control_pair_acc)),
                "named_pair_acc_gain_rank": int(1 + np.sum(control_pair_acc > named_pair_acc)),
                "named_margin_gain_mean": named_margin,
                "control_margin_gain_min": float(np.min(control_margin)),
                "control_margin_gain_max": float(np.max(control_margin)),
                "named_margin_gain_rank": int(1 + np.sum(control_margin > named_margin)),
            }
        )

    return pd.DataFrame(rows)


def plot_spatial_control_summary(
    gain_group_summary: pd.DataFrame,
    *,
    focus_region: str,
    outfile: Path,
) -> None:
    """Plot named-region and control hybrid gains for one focus window."""
    if gain_group_summary.empty:
        return

    plot_df = gain_group_summary.copy()
    display_name = np.where(
        plot_df["group_kind"] == "named_region",
        plot_df["region_name"],
        plot_df["region_name"].str.replace(f"{focus_region}_", "", regex=False),
    )
    plot_df["display_name"] = display_name
    plot_df["plot_order"] = np.where(plot_df["group_kind"] == "named_region", 0, 1)
    plot_df = plot_df.sort_values(["plot_order", "control_draw_idx", "region_name"]).reset_index(drop=True)

    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(plot_df.shape[0], dtype=float)
    colours = np.where(plot_df["group_kind"] == "named_region", "#c1121f", "#6c757d")

    axes[0].bar(x, plot_df["pair_acc_gain_mean"], color=colours, alpha=0.9)
    axes[0].errorbar(
        x,
        plot_df["pair_acc_gain_mean"],
        yerr=np.vstack(
            [
                plot_df["pair_acc_gain_mean"] - plot_df["pair_acc_gain_ci95_low"],
                plot_df["pair_acc_gain_ci95_high"] - plot_df["pair_acc_gain_mean"],
            ]
        ),
        fmt="none",
        ecolor="#1f2d3d",
        capsize=3,
    )
    axes[0].axhline(0.0, color="#495057", linestyle="--", linewidth=1.0)
    axes[0].set_title("Hybrid gain in matched-pair accuracy")
    axes[0].set_ylabel("Hybrid minus raw pair accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["display_name"], rotation=25, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, plot_df["mean_margin_gain_mean"], color=colours, alpha=0.9)
    axes[1].errorbar(
        x,
        plot_df["mean_margin_gain_mean"],
        yerr=np.vstack(
            [
                plot_df["mean_margin_gain_mean"] - plot_df["mean_margin_gain_ci95_low"],
                plot_df["mean_margin_gain_ci95_high"] - plot_df["mean_margin_gain_mean"],
            ]
        ),
        fmt="none",
        ecolor="#1f2d3d",
        capsize=3,
    )
    axes[1].axhline(0.0, color="#495057", linestyle="--", linewidth=1.0)
    axes[1].set_title("Hybrid gain in matched-pair score margin")
    axes[1].set_ylabel("Hybrid minus raw mean margin")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["display_name"], rotation=25, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_spatial_control_report(
    comparison_summary: pd.DataFrame,
    absolute_comparison_summary: pd.DataFrame,
    draw_rank_summary: pd.DataFrame,
    *,
    focus_region: str,
) -> str:
    """Write a short headline report for the named-versus-control comparison."""
    lines = [
        "# Matched Spatial-Control Summary",
        "",
        f"Focus region: {focus_region}",
        "",
    ]

    if not comparison_summary.empty:
        best_row = comparison_summary.sort_values(
            "mean_margin_gain_minus_control_mean",
            ascending=False,
        ).iloc[0]
        lines.extend(
            [
                "## Named region versus controls",
                "",
                (
                    f"- {best_row['matched_region_name']} at "
                    f"{int(best_row['target_start_ms'])} to {int(best_row['target_end_ms'])} ms: "
                    f"hybrid-minus-raw matched-pair accuracy gain exceeded the mean size-matched "
                    f"control gain by {best_row['pair_acc_gain_minus_control_mean']:.3f} "
                    f"(95% CI {best_row['pair_acc_gain_minus_control_ci95_low']:.3f} to "
                    f"{best_row['pair_acc_gain_minus_control_ci95_high']:.3f}, "
                    f"paired t p={best_row['pair_acc_gain_minus_control_ttest_p']:.3g}, "
                    f"Wilcoxon p={best_row['pair_acc_gain_minus_control_wilcoxon_p']:.3g})."
                ),
                (
                    f"- The same contrast in score margin was "
                    f"{best_row['mean_margin_gain_minus_control_mean']:.3f} "
                    f"(95% CI {best_row['mean_margin_gain_minus_control_ci95_low']:.3f} to "
                    f"{best_row['mean_margin_gain_minus_control_ci95_high']:.3f}, "
                    f"paired t p={best_row['mean_margin_gain_minus_control_ttest_p']:.3g}, "
                    f"Wilcoxon p={best_row['mean_margin_gain_minus_control_wilcoxon_p']:.3g})."
                ),
                "",
            ]
        )

    if not absolute_comparison_summary.empty:
        hybrid_rows = absolute_comparison_summary.loc[
            absolute_comparison_summary["model_name"] == DEFAULT_HYBRID_MODEL
        ].copy()
        if not hybrid_rows.empty:
            hybrid_row = hybrid_rows.iloc[0]
            lines.extend(
                [
                    "## Absolute hybrid performance versus controls",
                    "",
                    (
                        f"- Named {focus_region} hybrid matched-pair accuracy differed from the mean "
                        f"size-matched control accuracy by "
                        f"{hybrid_row['pair_acc_minus_control_mean']:.3f} "
                        f"(95% CI {hybrid_row['pair_acc_minus_control_ci95_low']:.3f} to "
                        f"{hybrid_row['pair_acc_minus_control_ci95_high']:.3f}, "
                        f"paired t p={hybrid_row['pair_acc_minus_control_ttest_p']:.3g}, "
                        f"Wilcoxon p={hybrid_row['pair_acc_minus_control_wilcoxon_p']:.3g})."
                    ),
                    (
                        f"- Named {focus_region} hybrid score margin differed from the mean control "
                        f"margin by {hybrid_row['mean_margin_minus_control_mean']:.3f} "
                        f"(95% CI {hybrid_row['mean_margin_minus_control_ci95_low']:.3f} to "
                        f"{hybrid_row['mean_margin_minus_control_ci95_high']:.3f}, "
                        f"paired t p={hybrid_row['mean_margin_minus_control_ttest_p']:.3g}, "
                        f"Wilcoxon p={hybrid_row['mean_margin_minus_control_wilcoxon_p']:.3g})."
                    ),
                    "",
                ]
            )

    if not draw_rank_summary.empty:
        rank_row = draw_rank_summary.iloc[0]
        lines.extend(
            [
                "## Control-draw range",
                "",
                (
                    f"- Named {focus_region} pair-accuracy gain ranked "
                    f"{int(rank_row['named_pair_acc_gain_rank'])} of "
                    f"{int(rank_row['n_control_draws']) + 1} against the control draws "
                    f"(control range {rank_row['control_pair_acc_gain_min']:.3f} to "
                    f"{rank_row['control_pair_acc_gain_max']:.3f})."
                ),
                (
                    f"- Named {focus_region} score-margin gain ranked "
                    f"{int(rank_row['named_margin_gain_rank'])} of "
                    f"{int(rank_row['n_control_draws']) + 1} against the control draws "
                    f"(control range {rank_row['control_margin_gain_min']:.3f} to "
                    f"{rank_row['control_margin_gain_max']:.3f})."
                ),
                "",
            ]
        )

    if len(lines) == 4:
        lines.append("No valid spatial-control rows were available.")
    return "\n".join(lines)


def run_matched_spatial_control_summary(
    *,
    baseline_results_dir: Path = DEFAULT_BASELINE_RESULTS_DIR,
    derivatives_dir: Path = DEFAULT_DERIVATIVES_DIR,
    export_dir: Path = DEFAULT_EXPORT_DIR,
    outdir: Path | None = None,
    rep_dim: int = DEFAULT_REP_DIM,
    focus_region: str = DEFAULT_FOCUS_REGION,
    target_start_ms: float = DEFAULT_TARGET_START_MS,
    target_end_ms: float = DEFAULT_TARGET_END_MS,
    positive_sdt: str = DEFAULT_POSITIVE_SDT,
    negative_sdt: str = DEFAULT_NEGATIVE_SDT,
    random_state: int = DEFAULT_RANDOM_SEED,
    n_bootstrap: int = DEFAULT_N_BOOT,
) -> int:
    """Run the matched hit-versus-miss spatial-control comparison."""
    outdir = (
        baseline_results_dir / "summary_matched_spatial_control"
        if outdir is None
        else outdir
    )
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        baseline_df = load_baseline_region_results(baseline_results_dir)
        focus_df = filter_focus_region_rows(
            baseline_df,
            rep_dim=rep_dim,
            focus_region=focus_region,
            target_start_ms=target_start_ms,
            target_end_ms=target_end_ms,
        )
        common_trials = collect_complete_trials(focus_df)
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
            raise ValueError("No hit or miss trials remained after aligning the spatial-control inputs")
        metadata_df = assign_subject_twofold_splits(metadata_df)

        n_control_draws = int(
            focus_df.loc[focus_df["group_kind"] == "size_matched_control", "control_draw_idx"]
            .dropna()
            .nunique()
        )
        if n_control_draws <= 0:
            raise ValueError("The selected baseline results do not contain size-matched controls")

        train_start_values = sorted(focus_df["train_start_ms"].dropna().astype(float).unique().tolist())
        train_end_values = sorted(focus_df["train_end_ms"].dropna().astype(float).unique().tolist())
        if len(train_start_values) != 1 or len(train_end_values) != 1:
            raise ValueError("The selected baseline rows do not share one fixed training window")
        train_start_ms = float(train_start_values[0])
        train_end_ms = float(train_end_values[0])

        reference_labels, subject_labels, subject_groups = build_subject_group_index_map(
            subjects,
            focus_region=focus_region,
            n_control_draws=n_control_draws,
            random_state=random_state,
            derivatives_dir=derivatives_dir,
        )
        raw_df = build_group_raw_feature_frame(
            common_trials,
            export_dir=export_dir,
            subject_labels=subject_labels,
            subject_groups=subject_groups,
            reference_labels=reference_labels,
            baseline_start_ms=train_start_ms,
            baseline_end_ms=train_end_ms,
            target_start_ms=target_start_ms,
            target_end_ms=target_end_ms,
        )

        trial_scores_frames = []
        group_rows = (
            focus_df[["group_kind", "matched_region_name", "region_name", "control_draw_idx"]]
            .drop_duplicates()
            .sort_values(["group_kind", "region_name"])
        )
        for row in group_rows.itertuples(index=False):
            group_kind = str(row.group_kind)
            matched_region_name = str(row.matched_region_name)
            region_name = str(row.region_name)
            control_draw_idx = row.control_draw_idx

            group_baseline_df = focus_df.loc[focus_df["region_name"] == region_name].copy()
            if group_baseline_df.empty:
                raise ValueError(f"No baseline rows were available for {region_name!r}")

            for model_name, model_label, model_kind, metrics in DEFAULT_MODEL_DEFS:
                feature_df = build_group_model_feature_frame(
                    model_kind,
                    metrics,
                    group_baseline_df=group_baseline_df,
                    raw_df=raw_df,
                    region_name=region_name,
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
                scores["group_kind"] = group_kind
                scores["matched_region_name"] = matched_region_name
                scores["region_name"] = region_name
                scores["control_draw_idx"] = control_draw_idx
                scores["target_start_ms"] = float(target_start_ms)
                scores["target_end_ms"] = float(target_end_ms)
                scores["target_center_ms"] = 0.5 * (
                    float(target_start_ms) + float(target_end_ms)
                )
                trial_scores_frames.append(scores)

        trial_scores = pd.concat(trial_scores_frames, ignore_index=True)
        trial_scores = merge_stimamp(trial_scores, derivatives_dir=derivatives_dir)
        pair_df = build_matched_pairs(
            trial_scores,
            positive_sdt=positive_sdt,
            negative_sdt=negative_sdt,
        )
        subject_pair_summary = evaluate_subject_pair_metrics_by_group(
            trial_scores,
            pair_df,
        )
        group_pair_summary = summarise_group_pair_metrics(
            subject_pair_summary,
            n_bootstrap=n_bootstrap,
            random_seed=random_state,
        )
        gain_subject_summary = build_hybrid_gain_subject_summary(
            subject_pair_summary,
            raw_model=DEFAULT_RAW_MODEL,
            hybrid_model=DEFAULT_HYBRID_MODEL,
        )
        gain_group_summary = summarise_hybrid_gain(
            gain_subject_summary,
            n_bootstrap=n_bootstrap,
            random_seed=random_state,
        )
        model_vs_control_subject = build_named_vs_control_model_subject_summary(
            subject_pair_summary
        )
        model_vs_control_group = summarise_named_vs_control_model(
            model_vs_control_subject,
            n_bootstrap=n_bootstrap,
            random_seed=random_state,
        )
        named_vs_control_subject = build_named_vs_control_subject_summary(
            gain_subject_summary
        )
        named_vs_control_group = summarise_named_vs_control(
            named_vs_control_subject,
            n_bootstrap=n_bootstrap,
            random_seed=random_state,
        )
        draw_rank_summary = build_control_draw_rank_summary(gain_group_summary)
        match_quality = build_match_quality_summary(pair_df)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    trial_scores_path = outdir / "matched_spatial_control_trial_scores.csv"
    pairs_path = outdir / "matched_spatial_control_pairs.csv"
    subject_pair_path = outdir / "matched_spatial_control_subject_summary.csv"
    group_pair_path = outdir / "matched_spatial_control_group_summary.csv"
    gain_subject_path = outdir / "matched_spatial_control_hybrid_gain_subject_summary.csv"
    gain_group_path = outdir / "matched_spatial_control_hybrid_gain_group_summary.csv"
    model_control_subject_path = outdir / "matched_spatial_control_model_vs_control_subject.csv"
    model_control_group_path = outdir / "matched_spatial_control_model_vs_control_group.csv"
    named_control_subject_path = outdir / "matched_spatial_control_named_vs_control_subject.csv"
    named_control_group_path = outdir / "matched_spatial_control_named_vs_control_group.csv"
    draw_rank_path = outdir / "matched_spatial_control_draw_ranks.csv"
    match_quality_path = outdir / "matched_spatial_control_match_quality.csv"
    plot_path = outdir / "matched_spatial_control_hybrid_gain.png"
    report_path = outdir / "matched_spatial_control_summary.md"

    save_table(trial_scores, trial_scores_path)
    save_table(pair_df, pairs_path)
    save_table(subject_pair_summary, subject_pair_path)
    save_table(group_pair_summary, group_pair_path)
    save_table(gain_subject_summary, gain_subject_path)
    save_table(gain_group_summary, gain_group_path)
    if not model_vs_control_subject.empty:
        save_table(model_vs_control_subject, model_control_subject_path)
    if not model_vs_control_group.empty:
        save_table(model_vs_control_group, model_control_group_path)
    if not named_vs_control_subject.empty:
        save_table(named_vs_control_subject, named_control_subject_path)
    if not named_vs_control_group.empty:
        save_table(named_vs_control_group, named_control_group_path)
    if not draw_rank_summary.empty:
        save_table(draw_rank_summary, draw_rank_path)
    save_table(match_quality, match_quality_path)
    plot_spatial_control_summary(
        gain_group_summary,
        focus_region=focus_region,
        outfile=plot_path,
    )
    report_path.write_text(
        build_spatial_control_report(
            named_vs_control_group,
            model_vs_control_group,
            draw_rank_summary,
            focus_region=focus_region,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved {trial_scores_path}")
    print(f"Saved {pairs_path}")
    print(f"Saved {subject_pair_path}")
    print(f"Saved {group_pair_path}")
    print(f"Saved {gain_subject_path}")
    print(f"Saved {gain_group_path}")
    if not model_vs_control_subject.empty:
        print(f"Saved {model_control_subject_path}")
    if not model_vs_control_group.empty:
        print(f"Saved {model_control_group_path}")
    if not named_vs_control_subject.empty:
        print(f"Saved {named_control_subject_path}")
    if not named_vs_control_group.empty:
        print(f"Saved {named_control_group_path}")
    if not draw_rank_summary.empty:
        print(f"Saved {draw_rank_path}")
    print(f"Saved {match_quality_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {report_path}")
    return 0
