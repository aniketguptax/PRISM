"""Summaries for the region-wise sliding baseline analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from framework.summaries import prepare_pyplot, save_table
from reporting.behaviour import DEFAULT_DERIVATIVES_DIR, load_all_epoch_metadata


DEFAULT_SDT_CONTRASTS = (("hit", "miss"), ("crej", "fa"))
DEFAULT_CONFIDENCE_SDTS = ("hit", "miss")
DEFAULT_MIN_CONFIDENCE_TRIALS = 20
DEFAULT_MIN_CONFIDENCE_UNIQUE_VALUES = 4
REQUIRED_REGION_COLUMNS = [
    "subject",
    "trial_idx",
    "region_name",
    "n_region_channels",
    "target_start_ms",
    "target_end_ms",
    "rep_dim",
    "pred_r2_obs",
    "pred_mse_obs",
]


def _split_balanced_confidence_groups(confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(confidence, kind="mergesort")
    half = order.size // 2
    return order[:half], order[-half:]


def _summarise_one_sample(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "positive_subject_share": float("nan"),
            "n_subjects": 0,
            "ttest_stat": float("nan"),
            "ttest_p": float("nan"),
        }

    if values.size >= 2:
        ttest_stat, ttest_p = stats.ttest_1samp(values, 0.0, nan_policy="omit")
        ttest_stat = float(ttest_stat)
        ttest_p = float(ttest_p)
    else:
        ttest_stat = float("nan")
        ttest_p = float("nan")

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "median": float(np.median(values)),
        "positive_subject_share": float(np.mean(values > 0.0)),
        "n_subjects": int(values.size),
        "ttest_stat": ttest_stat,
        "ttest_p": ttest_p,
    }


def load_region_sliding_results(results_dir: Path) -> pd.DataFrame:
    subject_files = sorted(results_dir.glob("sub-*_region_sliding_baseline.csv"))
    if subject_files:
        frames = [pd.read_csv(path) for path in subject_files]
        df = pd.concat(frames, ignore_index=True)
    else:
        combined_path = results_dir / "all_subjects_region_sliding_baseline.csv"
        if not combined_path.exists():
            raise FileNotFoundError(
                f"No region-sliding CSVs found in {results_dir}"
            )
        df = pd.read_csv(combined_path)

    missing = [column for column in REQUIRED_REGION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Region-sliding results are missing required columns: {missing}")

    df = df.copy()
    if "group_kind" not in df.columns:
        df["group_kind"] = "named_region"
    if "matched_region_name" not in df.columns:
        df["matched_region_name"] = df["region_name"]
    if "control_draw_idx" not in df.columns:
        df["control_draw_idx"] = np.nan
    if "train_mode" not in df.columns:
        df["train_mode"] = "rolling_history"
    if "baseline_duration_ms" not in df.columns:
        df["baseline_duration_ms"] = np.nan
    if "history_ms" not in df.columns:
        df["history_ms"] = np.nan
    if "target_duration_ms" not in df.columns:
        df["target_duration_ms"] = df["target_end_ms"] - df["target_start_ms"]
    if "step_ms" not in df.columns:
        df["step_ms"] = np.nan
    if "target_center_ms" not in df.columns:
        df["target_center_ms"] = 0.5 * (df["target_start_ms"] + df["target_end_ms"])
    if "error" in df.columns:
        df = df.loc[df["error"].fillna("").astype(str).eq("")].copy()

    return df.sort_values(
        ["subject", "trial_idx", "group_kind", "region_name", "target_start_ms", "rep_dim"]
    ).reset_index(drop=True)


def filter_rep_dim(df: pd.DataFrame, rep_dim: int | None) -> pd.DataFrame:
    if rep_dim is None:
        return df.copy()

    filtered = df.loc[df["rep_dim"] == int(rep_dim)].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for rep_dim={rep_dim}")
    return filtered


def merge_region_results_with_metadata(
    region_df: pd.DataFrame,
    *,
    derivatives_dir: Path,
) -> pd.DataFrame:
    subjects = sorted(region_df["subject"].dropna().astype(str).unique().tolist())
    metadata_df = load_all_epoch_metadata(subjects, derivatives_dir=derivatives_dir)
    return region_df.merge(
        metadata_df,
        on=["subject", "trial_idx"],
        how="left",
        validate="many_to_one",
    )


def build_region_subject_means(region_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "subject",
        "train_mode",
        "rep_dim",
        "group_kind",
        "matched_region_name",
        "region_name",
        "n_region_channels",
        "target_start_ms",
        "target_end_ms",
    ]
    subject_means = (
        region_df.groupby(group_columns, observed=False)
        .agg(
            control_draw_idx=("control_draw_idx", "first"),
            history_ms=("history_ms", "first"),
            baseline_duration_ms=("baseline_duration_ms", "first"),
            target_duration_ms=("target_duration_ms", "first"),
            step_ms=("step_ms", "first"),
            target_center_ms=("target_center_ms", "first"),
            pred_r2_obs=("pred_r2_obs", "mean"),
            pred_mse_obs=("pred_mse_obs", "mean"),
        )
        .reset_index()
    )
    return subject_means.sort_values(
        ["subject", "group_kind", "region_name", "target_start_ms", "rep_dim"]
    ).reset_index(drop=True)


def build_region_group_summary(subject_means: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "train_mode",
        "rep_dim",
        "group_kind",
        "matched_region_name",
        "region_name",
        "n_region_channels",
        "target_start_ms",
        "target_end_ms",
    ]
    summary = (
        subject_means.groupby(group_columns, observed=False)
        .agg(
            control_draw_idx=("control_draw_idx", "first"),
            history_ms=("history_ms", "first"),
            baseline_duration_ms=("baseline_duration_ms", "first"),
            target_duration_ms=("target_duration_ms", "first"),
            step_ms=("step_ms", "first"),
            target_center_ms=("target_center_ms", "first"),
            pred_r2_obs_mean=("pred_r2_obs", "mean"),
            pred_r2_obs_std=("pred_r2_obs", "std"),
            pred_r2_obs_median=("pred_r2_obs", "median"),
            pred_r2_obs_count=("pred_r2_obs", "count"),
            pred_mse_obs_mean=("pred_mse_obs", "mean"),
            pred_mse_obs_std=("pred_mse_obs", "std"),
            pred_mse_obs_median=("pred_mse_obs", "median"),
            pred_mse_obs_count=("pred_mse_obs", "count"),
        )
        .reset_index()
    )
    return summary.sort_values(
        ["group_kind", "region_name", "target_start_ms", "rep_dim"]
    ).reset_index(drop=True)


def build_group_condition_subject_deltas(
    merged_df: pd.DataFrame,
    *,
    contrasts: tuple[tuple[str, str], ...] = DEFAULT_SDT_CONTRASTS,
) -> pd.DataFrame:
    if merged_df.empty:
        return pd.DataFrame()

    analysis_df = merged_df.copy()
    analysis_df["_control_draw_group"] = (
        analysis_df["control_draw_idx"].fillna(0).astype(int)
    )
    group_key_columns = [
        "subject",
        "train_mode",
        "rep_dim",
        "group_kind",
        "matched_region_name",
        "region_name",
        "n_region_channels",
        "_control_draw_group",
        "target_start_ms",
        "target_end_ms",
    ]
    output_group_columns = [
        "subject",
        "train_mode",
        "rep_dim",
        "group_kind",
        "matched_region_name",
        "region_name",
        "n_region_channels",
        "control_draw_idx",
        "target_start_ms",
        "target_end_ms",
    ]
    metric_df = (
        analysis_df.groupby([*group_key_columns, "sdt"], observed=False)
        .agg(
            baseline_duration_ms=("baseline_duration_ms", "first"),
            history_ms=("history_ms", "first"),
            target_duration_ms=("target_duration_ms", "first"),
            target_center_ms=("target_center_ms", "first"),
            pred_r2_obs=("pred_r2_obs", "mean"),
        )
        .reset_index()
    )

    rows = []
    for condition_a, condition_b in contrasts:
        pair_df = metric_df.loc[metric_df["sdt"].isin([condition_a, condition_b])].copy()
        if pair_df.empty:
            continue

        wide = pair_df.pivot_table(
            index=group_key_columns,
            columns="sdt",
            values="pred_r2_obs",
            aggfunc="first",
        )
        wide = wide.dropna(subset=[condition_a, condition_b], how="any")
        if wide.empty:
            continue

        wide = wide.reset_index()
        wide = wide.rename(columns={"_control_draw_group": "control_draw_idx"})
        wide.loc[wide["group_kind"] == "named_region", "control_draw_idx"] = np.nan
        metadata = (
            pair_df.drop_duplicates(group_key_columns)[
                [
                    *group_key_columns,
                    "baseline_duration_ms",
                    "history_ms",
                    "target_duration_ms",
                    "target_center_ms",
                ]
            ]
        )
        metadata = metadata.rename(columns={"_control_draw_group": "control_draw_idx"})
        wide = wide.merge(
            metadata,
            on=output_group_columns,
            how="left",
            validate="one_to_one",
        )
        wide["contrast_name"] = f"{condition_a}_minus_{condition_b}"
        wide["condition_a"] = condition_a
        wide["condition_b"] = condition_b
        wide["delta_pred_r2_obs"] = wide[condition_a] - wide[condition_b]
        rows.append(
            wide[
                [
                    *output_group_columns,
                    "baseline_duration_ms",
                    "history_ms",
                    "target_duration_ms",
                    "target_center_ms",
                    "contrast_name",
                    "condition_a",
                    "condition_b",
                    "delta_pred_r2_obs",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(
        [
            "contrast_name",
            "group_kind",
            "matched_region_name",
            "region_name",
            "target_start_ms",
            "subject",
            "rep_dim",
        ]
    ).reset_index(drop=True)


def build_condition_subject_deltas(
    merged_df: pd.DataFrame,
    *,
    contrasts: tuple[tuple[str, str], ...] = DEFAULT_SDT_CONTRASTS,
) -> pd.DataFrame:
    group_deltas = build_group_condition_subject_deltas(
        merged_df,
        contrasts=contrasts,
    )
    if group_deltas.empty:
        return pd.DataFrame()
    return group_deltas.loc[group_deltas["group_kind"] == "named_region"].copy()


def build_group_confidence_subject_associations(
    merged_df: pd.DataFrame,
    *,
    sdts: tuple[str, ...] = DEFAULT_CONFIDENCE_SDTS,
    min_trials: int = DEFAULT_MIN_CONFIDENCE_TRIALS,
    min_unique_values: int = DEFAULT_MIN_CONFIDENCE_UNIQUE_VALUES,
) -> pd.DataFrame:
    if merged_df.empty or "confidence" not in merged_df.columns:
        return pd.DataFrame()

    analysis_df = merged_df.loc[merged_df["sdt"].isin(sdts)].copy()
    analysis_df = analysis_df.dropna(subset=["confidence", "pred_r2_obs"])
    if analysis_df.empty:
        return pd.DataFrame()

    analysis_df["_control_draw_group"] = (
        analysis_df["control_draw_idx"].fillna(0).astype(int)
    )
    group_key_columns = [
        "subject",
        "train_mode",
        "rep_dim",
        "group_kind",
        "matched_region_name",
        "region_name",
        "n_region_channels",
        "_control_draw_group",
        "target_start_ms",
        "target_end_ms",
        "sdt",
    ]

    rows = []
    for group_values, group_df in analysis_df.groupby(group_key_columns, observed=False):
        confidence = group_df["confidence"].to_numpy(dtype=float)
        pred_r2_obs = group_df["pred_r2_obs"].to_numpy(dtype=float)
        n_trials = int(confidence.size)
        n_unique_values = int(np.unique(confidence).size)
        if n_trials < min_trials or n_unique_values < min_unique_values:
            continue

        low_idx, high_idx = _split_balanced_confidence_groups(confidence)
        if low_idx.size == 0 or high_idx.size == 0:
            continue

        spearman_rho, spearman_p = stats.spearmanr(confidence, pred_r2_obs)
        if not np.isfinite(spearman_rho):
            continue

        row = dict(zip(group_key_columns, group_values))
        row["control_draw_idx"] = row.pop("_control_draw_group")
        if row["group_kind"] == "named_region":
            row["control_draw_idx"] = np.nan
        row.update(
            {
                "baseline_duration_ms": float(group_df["baseline_duration_ms"].iloc[0])
                if "baseline_duration_ms" in group_df
                else np.nan,
                "history_ms": float(group_df["history_ms"].iloc[0])
                if "history_ms" in group_df
                else np.nan,
                "target_duration_ms": float(group_df["target_duration_ms"].iloc[0]),
                "target_center_ms": float(group_df["target_center_ms"].iloc[0]),
                "n_trials": n_trials,
                "n_confidence_unique": n_unique_values,
                "confidence_mean": float(np.mean(confidence)),
                "confidence_std": float(np.std(confidence, ddof=1))
                if n_trials > 1
                else float("nan"),
                "pred_r2_obs_mean": float(np.mean(pred_r2_obs)),
                "confidence_pred_r2_spearman_rho": float(spearman_rho),
                "confidence_pred_r2_spearman_p": float(spearman_p),
                "confidence_high_minus_low_pred_r2_obs": float(
                    np.mean(pred_r2_obs[high_idx]) - np.mean(pred_r2_obs[low_idx])
                ),
                "confidence_low_trials": int(low_idx.size),
                "confidence_high_trials": int(high_idx.size),
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(
        [
            "sdt",
            "group_kind",
            "matched_region_name",
            "region_name",
            "target_start_ms",
            "subject",
            "rep_dim",
        ]
    ).reset_index(drop=True)


def build_confidence_subject_associations(
    merged_df: pd.DataFrame,
    *,
    sdts: tuple[str, ...] = DEFAULT_CONFIDENCE_SDTS,
    min_trials: int = DEFAULT_MIN_CONFIDENCE_TRIALS,
    min_unique_values: int = DEFAULT_MIN_CONFIDENCE_UNIQUE_VALUES,
) -> pd.DataFrame:
    group_associations = build_group_confidence_subject_associations(
        merged_df,
        sdts=sdts,
        min_trials=min_trials,
        min_unique_values=min_unique_values,
    )
    if group_associations.empty:
        return pd.DataFrame()
    return group_associations.loc[group_associations["group_kind"] == "named_region"].copy()


def summarise_confidence_associations(subject_associations: pd.DataFrame) -> pd.DataFrame:
    if subject_associations.empty:
        return pd.DataFrame()

    group_columns = [
        "train_mode",
        "rep_dim",
        "region_name",
        "n_region_channels",
        "target_start_ms",
        "target_end_ms",
        "sdt",
    ]
    rows = []
    for group_values, group_df in subject_associations.groupby(group_columns, observed=False):
        rho_summary = _summarise_one_sample(
            group_df["confidence_pred_r2_spearman_rho"].to_numpy(dtype=float)
        )
        delta_summary = _summarise_one_sample(
            group_df["confidence_high_minus_low_pred_r2_obs"].to_numpy(dtype=float)
        )
        row = dict(zip(group_columns, group_values))
        row.update(
            {
                "baseline_duration_ms": float(group_df["baseline_duration_ms"].iloc[0])
                if "baseline_duration_ms" in group_df
                else np.nan,
                "history_ms": float(group_df["history_ms"].iloc[0])
                if "history_ms" in group_df
                else np.nan,
                "target_duration_ms": float(group_df["target_duration_ms"].iloc[0]),
                "target_center_ms": float(group_df["target_center_ms"].iloc[0]),
                "mean_confidence_pred_r2_spearman_rho": rho_summary["mean"],
                "std_confidence_pred_r2_spearman_rho": rho_summary["std"],
                "median_confidence_pred_r2_spearman_rho": rho_summary["median"],
                "positive_rho_subject_share": rho_summary["positive_subject_share"],
                "n_subjects_rho": rho_summary["n_subjects"],
                "rho_ttest_stat": rho_summary["ttest_stat"],
                "rho_ttest_p": rho_summary["ttest_p"],
                "mean_confidence_high_minus_low_pred_r2_obs": delta_summary["mean"],
                "std_confidence_high_minus_low_pred_r2_obs": delta_summary["std"],
                "median_confidence_high_minus_low_pred_r2_obs": delta_summary["median"],
                "positive_high_low_subject_share": delta_summary["positive_subject_share"],
                "n_subjects_high_low": delta_summary["n_subjects"],
                "high_low_ttest_stat": delta_summary["ttest_stat"],
                "high_low_ttest_p": delta_summary["ttest_p"],
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["sdt", "region_name", "target_start_ms", "rep_dim"]
    ).reset_index(drop=True)


def summarise_condition_deltas(subject_deltas: pd.DataFrame) -> pd.DataFrame:
    if subject_deltas.empty:
        return pd.DataFrame()

    group_columns = [
        "train_mode",
        "rep_dim",
        "region_name",
        "n_region_channels",
        "target_start_ms",
        "target_end_ms",
        "contrast_name",
        "condition_a",
        "condition_b",
    ]

    rows = []
    for group_values, group_df in subject_deltas.groupby(group_columns, observed=False):
        values = group_df["delta_pred_r2_obs"].to_numpy(dtype=float)
        if values.size >= 2:
            ttest_stat, ttest_p = stats.ttest_1samp(values, 0.0, nan_policy="omit")
            ttest_stat = float(ttest_stat)
            ttest_p = float(ttest_p)
        else:
            ttest_stat = float("nan")
            ttest_p = float("nan")
        row = dict(zip(group_columns, group_values))
        row.update(
            {
                "baseline_duration_ms": float(group_df["baseline_duration_ms"].iloc[0])
                if "baseline_duration_ms" in group_df
                else np.nan,
                "history_ms": float(group_df["history_ms"].iloc[0])
                if "history_ms" in group_df
                else np.nan,
                "target_duration_ms": float(group_df["target_duration_ms"].iloc[0]),
                "target_center_ms": float(group_df["target_center_ms"].iloc[0]),
                "mean_delta_pred_r2_obs": float(np.mean(values)),
                "std_delta_pred_r2_obs": (
                    float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
                ),
                "median_delta_pred_r2_obs": float(np.median(values)),
                "positive_subject_share": float(np.mean(values > 0.0)),
                "n_subjects": int(values.size),
                "ttest_stat": ttest_stat,
                "ttest_p": ttest_p,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["contrast_name", "region_name", "target_start_ms", "rep_dim"]
    ).reset_index(drop=True)


def build_confidence_control_subject_deltas(
    group_confidence_subject_associations: pd.DataFrame,
) -> pd.DataFrame:
    if group_confidence_subject_associations.empty:
        return pd.DataFrame()

    control_df = group_confidence_subject_associations.loc[
        group_confidence_subject_associations["group_kind"] == "size_matched_control"
    ].copy()
    if control_df.empty:
        return pd.DataFrame()

    named_df = group_confidence_subject_associations.loc[
        (group_confidence_subject_associations["group_kind"] == "named_region")
        & group_confidence_subject_associations["region_name"].ne("all_channels")
    ].copy()
    if named_df.empty:
        return pd.DataFrame()

    merge_keys = [
        "subject",
        "train_mode",
        "rep_dim",
        "matched_region_name",
        "target_start_ms",
        "target_end_ms",
        "sdt",
    ]
    control_means = (
        control_df.groupby(merge_keys, observed=False)
        .agg(
            baseline_duration_ms=("baseline_duration_ms", "first"),
            history_ms=("history_ms", "first"),
            target_duration_ms=("target_duration_ms", "first"),
            target_center_ms=("target_center_ms", "first"),
            n_region_channels=("n_region_channels", "first"),
            control_mean_confidence_pred_r2_spearman_rho=(
                "confidence_pred_r2_spearman_rho",
                "mean",
            ),
            control_mean_confidence_high_minus_low_pred_r2_obs=(
                "confidence_high_minus_low_pred_r2_obs",
                "mean",
            ),
            n_control_draws=("control_draw_idx", "nunique"),
        )
        .reset_index()
    )
    named_means = named_df.rename(
        columns={
            "confidence_pred_r2_spearman_rho": "named_confidence_pred_r2_spearman_rho",
            "confidence_high_minus_low_pred_r2_obs": (
                "named_confidence_high_minus_low_pred_r2_obs"
            ),
        }
    )[
        [
            "subject",
            "train_mode",
            "rep_dim",
            "matched_region_name",
            "target_start_ms",
            "target_end_ms",
            "sdt",
            "named_confidence_pred_r2_spearman_rho",
            "named_confidence_high_minus_low_pred_r2_obs",
        ]
    ]
    merged = named_means.merge(
        control_means,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    merged["rho_minus_control_confidence_pred_r2_spearman_rho"] = (
        merged["named_confidence_pred_r2_spearman_rho"]
        - merged["control_mean_confidence_pred_r2_spearman_rho"]
    )
    merged["high_low_minus_control_pred_r2_obs"] = (
        merged["named_confidence_high_minus_low_pred_r2_obs"]
        - merged["control_mean_confidence_high_minus_low_pred_r2_obs"]
    )
    return merged.sort_values(
        ["sdt", "matched_region_name", "target_start_ms", "subject", "rep_dim"]
    ).reset_index(drop=True)


def summarise_confidence_control_deltas(
    confidence_control_subject_deltas: pd.DataFrame,
) -> pd.DataFrame:
    if confidence_control_subject_deltas.empty:
        return pd.DataFrame()

    group_columns = [
        "train_mode",
        "rep_dim",
        "matched_region_name",
        "n_region_channels",
        "target_start_ms",
        "target_end_ms",
        "sdt",
    ]
    rows = []
    for group_values, group_df in confidence_control_subject_deltas.groupby(
        group_columns,
        observed=False,
    ):
        rho_summary = _summarise_one_sample(
            group_df["rho_minus_control_confidence_pred_r2_spearman_rho"].to_numpy(dtype=float)
        )
        delta_summary = _summarise_one_sample(
            group_df["high_low_minus_control_pred_r2_obs"].to_numpy(dtype=float)
        )
        row = dict(zip(group_columns, group_values))
        row.update(
            {
                "baseline_duration_ms": float(group_df["baseline_duration_ms"].iloc[0]),
                "history_ms": float(group_df["history_ms"].iloc[0]),
                "target_duration_ms": float(group_df["target_duration_ms"].iloc[0]),
                "target_center_ms": float(group_df["target_center_ms"].iloc[0]),
                "n_control_draws": int(group_df["n_control_draws"].iloc[0]),
                "mean_rho_minus_control_confidence_pred_r2_spearman_rho": rho_summary["mean"],
                "std_rho_minus_control_confidence_pred_r2_spearman_rho": rho_summary["std"],
                "median_rho_minus_control_confidence_pred_r2_spearman_rho": rho_summary[
                    "median"
                ],
                "positive_rho_minus_control_subject_share": rho_summary[
                    "positive_subject_share"
                ],
                "n_subjects_rho": rho_summary["n_subjects"],
                "rho_ttest_stat": rho_summary["ttest_stat"],
                "rho_ttest_p": rho_summary["ttest_p"],
                "mean_high_low_minus_control_pred_r2_obs": delta_summary["mean"],
                "std_high_low_minus_control_pred_r2_obs": delta_summary["std"],
                "median_high_low_minus_control_pred_r2_obs": delta_summary["median"],
                "positive_high_low_minus_control_subject_share": delta_summary[
                    "positive_subject_share"
                ],
                "n_subjects_high_low": delta_summary["n_subjects"],
                "high_low_ttest_stat": delta_summary["ttest_stat"],
                "high_low_ttest_p": delta_summary["ttest_p"],
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["sdt", "matched_region_name", "target_start_ms", "rep_dim"]
    ).reset_index(drop=True)


def build_condition_control_subject_deltas(
    group_condition_deltas: pd.DataFrame,
) -> pd.DataFrame:
    if group_condition_deltas.empty:
        return pd.DataFrame()

    control_df = group_condition_deltas.loc[
        group_condition_deltas["group_kind"] == "size_matched_control"
    ].copy()
    if control_df.empty:
        return pd.DataFrame()

    named_df = group_condition_deltas.loc[
        (group_condition_deltas["group_kind"] == "named_region")
        & group_condition_deltas["region_name"].ne("all_channels")
    ].copy()
    if named_df.empty:
        return pd.DataFrame()

    merge_keys = [
        "subject",
        "train_mode",
        "rep_dim",
        "matched_region_name",
        "target_start_ms",
        "target_end_ms",
        "contrast_name",
        "condition_a",
        "condition_b",
    ]
    control_means = (
        control_df.groupby(merge_keys, observed=False)
        .agg(
            baseline_duration_ms=("baseline_duration_ms", "first"),
            history_ms=("history_ms", "first"),
            target_duration_ms=("target_duration_ms", "first"),
            target_center_ms=("target_center_ms", "first"),
            n_region_channels=("n_region_channels", "first"),
            control_mean_delta_pred_r2_obs=("delta_pred_r2_obs", "mean"),
            n_control_draws=("control_draw_idx", "nunique"),
        )
        .reset_index()
    )
    named_means = named_df.rename(
        columns={"delta_pred_r2_obs": "named_delta_pred_r2_obs"}
    )[
        [
            "subject",
            "train_mode",
            "rep_dim",
            "matched_region_name",
            "target_start_ms",
            "target_end_ms",
            "contrast_name",
            "condition_a",
            "condition_b",
            "named_delta_pred_r2_obs",
        ]
    ]
    merged = named_means.merge(
        control_means,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    merged["delta_minus_control_delta_pred_r2_obs"] = (
        merged["named_delta_pred_r2_obs"] - merged["control_mean_delta_pred_r2_obs"]
    )
    return merged.sort_values(
        ["contrast_name", "matched_region_name", "target_start_ms", "subject", "rep_dim"]
    ).reset_index(drop=True)


def summarise_condition_control_deltas(
    condition_control_subject_deltas: pd.DataFrame,
) -> pd.DataFrame:
    if condition_control_subject_deltas.empty:
        return pd.DataFrame()

    group_columns = [
        "train_mode",
        "rep_dim",
        "matched_region_name",
        "n_region_channels",
        "target_start_ms",
        "target_end_ms",
        "contrast_name",
        "condition_a",
        "condition_b",
    ]
    rows = []
    for group_values, group_df in condition_control_subject_deltas.groupby(
        group_columns,
        observed=False,
    ):
        values = group_df["delta_minus_control_delta_pred_r2_obs"].to_numpy(dtype=float)
        if values.size >= 2:
            ttest_stat, ttest_p = stats.ttest_1samp(values, 0.0, nan_policy="omit")
            ttest_stat = float(ttest_stat)
            ttest_p = float(ttest_p)
        else:
            ttest_stat = float("nan")
            ttest_p = float("nan")

        row = dict(zip(group_columns, group_values))
        row.update(
            {
                "baseline_duration_ms": float(group_df["baseline_duration_ms"].iloc[0]),
                "history_ms": float(group_df["history_ms"].iloc[0]),
                "target_duration_ms": float(group_df["target_duration_ms"].iloc[0]),
                "target_center_ms": float(group_df["target_center_ms"].iloc[0]),
                "n_control_draws": int(group_df["n_control_draws"].iloc[0]),
                "mean_delta_minus_control_delta_pred_r2_obs": float(np.mean(values)),
                "std_delta_minus_control_delta_pred_r2_obs": (
                    float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
                ),
                "median_delta_minus_control_delta_pred_r2_obs": float(np.median(values)),
                "positive_subject_share": float(np.mean(values > 0.0)),
                "n_subjects": int(values.size),
                "ttest_stat": ttest_stat,
                "ttest_p": ttest_p,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["contrast_name", "matched_region_name", "target_start_ms", "rep_dim"]
    ).reset_index(drop=True)


def build_control_subject_deltas(subject_means: pd.DataFrame) -> pd.DataFrame:
    control_df = subject_means.loc[subject_means["group_kind"] == "size_matched_control"].copy()
    if control_df.empty:
        return pd.DataFrame()

    named_df = subject_means.loc[
        (subject_means["group_kind"] == "named_region")
        & subject_means["region_name"].ne("all_channels")
    ].copy()
    if named_df.empty:
        return pd.DataFrame()

    merge_keys = [
        "subject",
        "train_mode",
        "rep_dim",
        "matched_region_name",
        "target_start_ms",
        "target_end_ms",
    ]
    control_means = (
        control_df.groupby(merge_keys, observed=False)
        .agg(
            history_ms=("history_ms", "first"),
            baseline_duration_ms=("baseline_duration_ms", "first"),
            target_duration_ms=("target_duration_ms", "first"),
            target_center_ms=("target_center_ms", "first"),
            n_region_channels=("n_region_channels", "first"),
            control_mean_pred_r2_obs=("pred_r2_obs", "mean"),
            n_control_draws=("control_draw_idx", "nunique"),
        )
        .reset_index()
    )
    named_means = named_df.rename(columns={"pred_r2_obs": "named_pred_r2_obs"})[
        [
            "subject",
            "train_mode",
            "rep_dim",
            "matched_region_name",
            "target_start_ms",
            "target_end_ms",
            "named_pred_r2_obs",
        ]
    ]
    merged = named_means.merge(
        control_means,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    merged["delta_pred_r2_obs"] = (
        merged["named_pred_r2_obs"] - merged["control_mean_pred_r2_obs"]
    )
    return merged.sort_values(
        ["matched_region_name", "target_start_ms", "subject", "rep_dim"]
    ).reset_index(drop=True)


def summarise_control_deltas(control_subject_deltas: pd.DataFrame) -> pd.DataFrame:
    if control_subject_deltas.empty:
        return pd.DataFrame()

    group_columns = [
        "train_mode",
        "rep_dim",
        "matched_region_name",
        "n_region_channels",
        "target_start_ms",
        "target_end_ms",
    ]
    rows = []
    for group_values, group_df in control_subject_deltas.groupby(group_columns, observed=False):
        values = group_df["delta_pred_r2_obs"].to_numpy(dtype=float)
        if values.size >= 2:
            ttest_stat, ttest_p = stats.ttest_1samp(values, 0.0, nan_policy="omit")
            ttest_stat = float(ttest_stat)
            ttest_p = float(ttest_p)
        else:
            ttest_stat = float("nan")
            ttest_p = float("nan")
        row = dict(zip(group_columns, group_values))
        row.update(
            {
                "history_ms": float(group_df["history_ms"].iloc[0]),
                "baseline_duration_ms": float(group_df["baseline_duration_ms"].iloc[0]),
                "target_duration_ms": float(group_df["target_duration_ms"].iloc[0]),
                "target_center_ms": float(group_df["target_center_ms"].iloc[0]),
                "n_control_draws": int(group_df["n_control_draws"].iloc[0]),
                "mean_delta_pred_r2_obs": float(np.mean(values)),
                "std_delta_pred_r2_obs": (
                    float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
                ),
                "median_delta_pred_r2_obs": float(np.median(values)),
                "positive_subject_share": float(np.mean(values > 0.0)),
                "n_subjects": int(values.size),
                "ttest_stat": ttest_stat,
                "ttest_p": ttest_p,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["matched_region_name", "target_start_ms", "rep_dim"]
    ).reset_index(drop=True)


def plot_region_timecourses(group_summary: pd.DataFrame, outfile: Path) -> None:
    plot_df = group_summary.loc[group_summary["group_kind"] == "named_region"].copy()
    if plot_df.empty:
        return

    train_modes = plot_df["train_mode"].dropna().astype(str).unique().tolist()
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(
        1,
        len(train_modes),
        figsize=(6 * len(train_modes), 4.5),
        constrained_layout=True,
    )
    if len(train_modes) == 1:
        axes = [axes]

    for axis, train_mode in zip(axes, train_modes):
        mode_df = plot_df.loc[plot_df["train_mode"] == train_mode].copy()
        for region_name, region_df in mode_df.groupby("region_name", observed=False):
            region_df = region_df.sort_values("target_start_ms")
            linestyle = "--" if region_name == "all_channels" else "-"
            linewidth = 2.6 if region_name != "all_channels" else 2.0
            axis.plot(
                region_df["target_start_ms"],
                region_df["pred_r2_obs_mean"],
                linestyle,
                linewidth=linewidth,
                label=region_name,
            )

        axis.axvline(0.0, color="#adb5bd", linewidth=1.0)
        axis.axhline(0.0, color="#495057", linewidth=1.0, linestyle="--")
        axis.set_title(train_mode.replace("_", " "))
        axis.set_xlabel("Target window start (ms)")
        axis.set_ylabel("Mean observed R^2")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_condition_contrasts(condition_summary: pd.DataFrame, outfile: Path) -> None:
    if condition_summary.empty:
        return

    train_modes = condition_summary["train_mode"].dropna().astype(str).unique().tolist()
    contrasts = condition_summary["contrast_name"].dropna().astype(str).unique().tolist()
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(
        len(train_modes),
        len(contrasts),
        figsize=(5.5 * len(contrasts), 4.2 * len(train_modes)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, train_mode in enumerate(train_modes):
        for col_idx, contrast_name in enumerate(contrasts):
            axis = axes[row_idx][col_idx]
            panel_df = condition_summary.loc[
                (condition_summary["train_mode"] == train_mode)
                & (condition_summary["contrast_name"] == contrast_name)
            ].copy()
            for region_name, region_df in panel_df.groupby("region_name", observed=False):
                region_df = region_df.sort_values("target_start_ms")
                axis.plot(
                    region_df["target_start_ms"],
                    region_df["mean_delta_pred_r2_obs"],
                    linewidth=2.4,
                    label=region_name,
                )

            axis.axvline(0.0, color="#adb5bd", linewidth=1.0)
            axis.axhline(0.0, color="#495057", linewidth=1.0, linestyle="--")
            axis.set_title(f"{train_mode.replace('_', ' ')} | {contrast_name}")
            axis.set_xlabel("Target window start (ms)")
            axis.set_ylabel("Condition delta in observed R^2")
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_associations(confidence_summary: pd.DataFrame, outfile: Path) -> None:
    if confidence_summary.empty:
        return

    plot_df = confidence_summary.loc[confidence_summary["region_name"] != "all_channels"].copy()
    if plot_df.empty:
        return

    train_modes = plot_df["train_mode"].dropna().astype(str).unique().tolist()
    sdts = plot_df["sdt"].dropna().astype(str).unique().tolist()
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(
        len(train_modes),
        len(sdts),
        figsize=(5.5 * len(sdts), 4.2 * len(train_modes)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, train_mode in enumerate(train_modes):
        for col_idx, sdt in enumerate(sdts):
            axis = axes[row_idx][col_idx]
            panel_df = plot_df.loc[
                (plot_df["train_mode"] == train_mode) & (plot_df["sdt"] == sdt)
            ].copy()
            for region_name, region_df in panel_df.groupby("region_name", observed=False):
                region_df = region_df.sort_values("target_start_ms")
                axis.plot(
                    region_df["target_start_ms"],
                    region_df["mean_confidence_pred_r2_spearman_rho"],
                    linewidth=2.4,
                    label=region_name,
                )

            axis.axvline(0.0, color="#adb5bd", linewidth=1.0)
            axis.axhline(0.0, color="#495057", linewidth=1.0, linestyle="--")
            axis.set_title(f"{train_mode.replace('_', ' ')} | {sdt}")
            axis.set_xlabel("Target window start (ms)")
            axis.set_ylabel("Mean subject Spearman rho")
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_region_report(
    group_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    condition_control_summary: pd.DataFrame,
    confidence_summary: pd.DataFrame,
    confidence_control_summary: pd.DataFrame,
) -> str:
    named_df = group_summary.loc[
        (group_summary["group_kind"] == "named_region")
        & group_summary["region_name"].ne("all_channels")
    ].copy()

    lines = [
        "# Region Sliding Summary",
        "",
    ]

    if not named_df.empty:
        best_region = named_df.sort_values("pred_r2_obs_mean", ascending=False).iloc[0]
        lines.extend(
            [
                "## Peak regional fit",
                "",
                (
                    f"- Best mean observed R^2: {best_region['region_name']} at "
                    f"{int(best_region['target_start_ms'])} to {int(best_region['target_end_ms'])} ms "
                    f"(q={int(best_region['rep_dim'])}, train={best_region['train_mode']}, "
                    f"mean={best_region['pred_r2_obs_mean']:.3f})."
                ),
                "",
            ]
        )

    if not condition_summary.empty:
        lines.extend(["## Condition contrasts", ""])
        for contrast_name, contrast_df in condition_summary.groupby("contrast_name", observed=False):
            best_row = contrast_df.sort_values("mean_delta_pred_r2_obs", ascending=False).iloc[0]
            lines.append(
                f"- {contrast_name}: strongest positive delta in {best_row['region_name']} at "
                f"{int(best_row['target_start_ms'])} to {int(best_row['target_end_ms'])} ms "
                f"(q={int(best_row['rep_dim'])}, mean delta={best_row['mean_delta_pred_r2_obs']:.3f}, "
                f"p={best_row['ttest_p']:.3g})."
            )
        lines.append("")

    if not control_summary.empty:
        best_control = control_summary.sort_values("mean_delta_pred_r2_obs", ascending=False).iloc[0]
        lines.extend(
            [
                "## Size-matched controls",
                "",
                (
                    f"- Strongest named-region advantage over the random size-matched controls: "
                    f"{best_control['matched_region_name']} at "
                    f"{int(best_control['target_start_ms'])} to {int(best_control['target_end_ms'])} ms "
                    f"(q={int(best_control['rep_dim'])}, mean delta={best_control['mean_delta_pred_r2_obs']:.3f}, "
                    f"p={best_control['ttest_p']:.3g})."
                ),
                "",
            ]
        )

    if not condition_control_summary.empty:
        best_condition_control = condition_control_summary.sort_values(
            "mean_delta_minus_control_delta_pred_r2_obs",
            ascending=False,
        ).iloc[0]
        lines.extend(
            [
                "## Condition deltas versus controls",
                "",
                (
                    f"- Strongest named-region advantage over the size-matched controls for the "
                    f"condition contrast itself: {best_condition_control['matched_region_name']} "
                    f"{best_condition_control['contrast_name']} at "
                    f"{int(best_condition_control['target_start_ms'])} to "
                    f"{int(best_condition_control['target_end_ms'])} ms "
                    f"(q={int(best_condition_control['rep_dim'])}, mean delta-over-control="
                    f"{best_condition_control['mean_delta_minus_control_delta_pred_r2_obs']:.3f}, "
                    f"p={best_condition_control['ttest_p']:.3g})."
                ),
                "",
            ]
        )

    if not confidence_summary.empty:
        named_confidence = confidence_summary.loc[
            confidence_summary["region_name"].ne("all_channels")
        ].copy()
        if not named_confidence.empty:
            lines.extend(["## Confidence associations", ""])
            for sdt, sdt_df in named_confidence.groupby("sdt", observed=False):
                best_row = sdt_df.sort_values(
                    "mean_confidence_pred_r2_spearman_rho",
                    ascending=False,
                ).iloc[0]
                lines.append(
                    f"- {sdt}: strongest positive confidence-fit association in "
                    f"{best_row['region_name']} at {int(best_row['target_start_ms'])} to "
                    f"{int(best_row['target_end_ms'])} ms "
                    f"(q={int(best_row['rep_dim'])}, mean rho="
                    f"{best_row['mean_confidence_pred_r2_spearman_rho']:.3f}, "
                    f"p={best_row['rho_ttest_p']:.3g})."
                )
            lines.append("")

    if not confidence_control_summary.empty:
        lines.extend(["## Confidence associations versus controls", ""])
        for sdt, sdt_df in confidence_control_summary.groupby("sdt", observed=False):
            best_row = sdt_df.sort_values(
                "mean_rho_minus_control_confidence_pred_r2_spearman_rho",
                ascending=False,
            ).iloc[0]
            lines.append(
                f"- {sdt}: strongest named-region advantage over size-matched controls in "
                f"confidence-fit association was {best_row['matched_region_name']} at "
                f"{int(best_row['target_start_ms'])} to {int(best_row['target_end_ms'])} ms "
                f"(q={int(best_row['rep_dim'])}, mean rho-over-control="
                f"{best_row['mean_rho_minus_control_confidence_pred_r2_spearman_rho']:.3f}, "
                f"p={best_row['rho_ttest_p']:.3g})."
            )
        lines.append("")

    if len(lines) == 2:
        lines.extend(["No valid rows were available for summary."])
    return "\n".join(lines)


def run_region_sliding_summary(
    *,
    results_dir: Path,
    derivatives_dir: Path = DEFAULT_DERIVATIVES_DIR,
    outdir: Path | None = None,
    rep_dim: int | None = None,
) -> int:
    outdir = results_dir / "summary" if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        region_df = load_region_sliding_results(results_dir)
        region_df = filter_rep_dim(region_df, rep_dim)
        merged_df = merge_region_results_with_metadata(
            region_df,
            derivatives_dir=derivatives_dir,
        )
        subject_means = build_region_subject_means(region_df)
        group_summary = build_region_group_summary(subject_means)
        group_condition_deltas = build_group_condition_subject_deltas(merged_df)
        condition_subject_deltas = build_condition_subject_deltas(merged_df)
        condition_summary = summarise_condition_deltas(condition_subject_deltas)
        group_confidence_subject_associations = build_group_confidence_subject_associations(
            merged_df
        )
        confidence_subject_associations = build_confidence_subject_associations(merged_df)
        confidence_summary = summarise_confidence_associations(
            confidence_subject_associations
        )
        control_subject_deltas = build_control_subject_deltas(subject_means)
        control_summary = summarise_control_deltas(control_subject_deltas)
        condition_control_subject_deltas = build_condition_control_subject_deltas(
            group_condition_deltas
        )
        condition_control_summary = summarise_condition_control_deltas(
            condition_control_subject_deltas
        )
        confidence_control_subject_deltas = build_confidence_control_subject_deltas(
            group_confidence_subject_associations
        )
        confidence_control_summary = summarise_confidence_control_deltas(
            confidence_control_subject_deltas
        )
        report = build_region_report(
            group_summary,
            condition_summary,
            control_summary,
            condition_control_summary,
            confidence_summary,
            confidence_control_summary,
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    suffix = "" if rep_dim is None else f"_q{int(rep_dim)}"
    subject_means_path = outdir / f"region_sliding_subject_means{suffix}.csv"
    group_summary_path = outdir / f"region_sliding_group_summary{suffix}.csv"
    condition_subject_path = outdir / f"region_sliding_condition_subject_deltas{suffix}.csv"
    condition_summary_path = outdir / f"region_sliding_condition_group_summary{suffix}.csv"
    confidence_subject_path = outdir / f"region_sliding_confidence_subject_associations{suffix}.csv"
    confidence_summary_path = outdir / f"region_sliding_confidence_group_summary{suffix}.csv"
    control_subject_path = outdir / f"region_sliding_control_subject_deltas{suffix}.csv"
    control_summary_path = outdir / f"region_sliding_control_group_summary{suffix}.csv"
    condition_control_subject_path = (
        outdir / f"region_sliding_condition_vs_control_subject_deltas{suffix}.csv"
    )
    condition_control_summary_path = (
        outdir / f"region_sliding_condition_vs_control_group_summary{suffix}.csv"
    )
    confidence_control_subject_path = (
        outdir / f"region_sliding_confidence_vs_control_subject_deltas{suffix}.csv"
    )
    confidence_control_summary_path = (
        outdir / f"region_sliding_confidence_vs_control_group_summary{suffix}.csv"
    )
    timecourse_plot_path = outdir / f"region_sliding_timecourses{suffix}.png"
    contrast_plot_path = outdir / f"region_sliding_condition_contrasts{suffix}.png"
    confidence_plot_path = outdir / f"region_sliding_confidence_associations{suffix}.png"
    report_path = outdir / f"region_sliding_summary{suffix}.md"

    save_table(subject_means, subject_means_path)
    save_table(group_summary, group_summary_path)
    save_table(condition_subject_deltas, condition_subject_path)
    save_table(condition_summary, condition_summary_path)
    if not confidence_subject_associations.empty:
        save_table(confidence_subject_associations, confidence_subject_path)
    if not confidence_summary.empty:
        save_table(confidence_summary, confidence_summary_path)
    if not control_subject_deltas.empty:
        save_table(control_subject_deltas, control_subject_path)
    if not control_summary.empty:
        save_table(control_summary, control_summary_path)
    if not condition_control_subject_deltas.empty:
        save_table(condition_control_subject_deltas, condition_control_subject_path)
    if not condition_control_summary.empty:
        save_table(condition_control_summary, condition_control_summary_path)
    if not confidence_control_subject_deltas.empty:
        save_table(confidence_control_subject_deltas, confidence_control_subject_path)
    if not confidence_control_summary.empty:
        save_table(confidence_control_summary, confidence_control_summary_path)
    plot_region_timecourses(group_summary, timecourse_plot_path)
    plot_condition_contrasts(condition_summary, contrast_plot_path)
    plot_confidence_associations(confidence_summary, confidence_plot_path)
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"Saved {subject_means_path}")
    print(f"Saved {group_summary_path}")
    print(f"Saved {condition_subject_path}")
    print(f"Saved {condition_summary_path}")
    if not confidence_subject_associations.empty:
        print(f"Saved {confidence_subject_path}")
    if not confidence_summary.empty:
        print(f"Saved {confidence_summary_path}")
    if not control_subject_deltas.empty:
        print(f"Saved {control_subject_path}")
    if not control_summary.empty:
        print(f"Saved {control_summary_path}")
    if not condition_control_subject_deltas.empty:
        print(f"Saved {condition_control_subject_path}")
    if not condition_control_summary.empty:
        print(f"Saved {condition_control_summary_path}")
    if not confidence_control_subject_deltas.empty:
        print(f"Saved {confidence_control_subject_path}")
    if not confidence_control_summary.empty:
        print(f"Saved {confidence_control_summary_path}")
    print(f"Saved {timecourse_plot_path}")
    print(f"Saved {contrast_plot_path}")
    print(f"Saved {confidence_plot_path}")
    print(f"Saved {report_path}")

    print("\nNamed-region peak rows:")
    named_summary = group_summary.loc[group_summary["group_kind"] == "named_region"].copy()
    print(
        named_summary.sort_values("pred_r2_obs_mean", ascending=False)
        .head(10)
        .to_string(index=False)
    )
    if not condition_summary.empty:
        print("\nCondition contrasts:")
        print(
            condition_summary.sort_values("mean_delta_pred_r2_obs", ascending=False)
            .head(10)
            .to_string(index=False)
        )
    if not confidence_summary.empty:
        print("\nConfidence associations:")
        print(
            confidence_summary.sort_values(
                "mean_confidence_pred_r2_spearman_rho",
                ascending=False,
            )
            .head(10)
            .to_string(index=False)
        )
    if not control_summary.empty:
        print("\nSize-matched control contrasts:")
        print(
            control_summary.sort_values("mean_delta_pred_r2_obs", ascending=False)
            .head(10)
            .to_string(index=False)
        )
    if not condition_control_summary.empty:
        print("\nCondition deltas versus size-matched controls:")
        print(
            condition_control_summary.sort_values(
                "mean_delta_minus_control_delta_pred_r2_obs",
                ascending=False,
            )
            .head(10)
            .to_string(index=False)
        )
    if not confidence_control_summary.empty:
        print("\nConfidence associations versus size-matched controls:")
        print(
            confidence_control_summary.sort_values(
                "mean_rho_minus_control_confidence_pred_r2_spearman_rho",
                ascending=False,
            )
            .head(10)
            .to_string(index=False)
        )
    return 0
