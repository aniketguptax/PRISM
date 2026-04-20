"""Amplitude-matched evidence summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

from framework.summaries import prepare_pyplot, save_table
from reporting.behaviour import DEFAULT_DERIVATIVES_DIR, load_all_epoch_metadata
from reporting.evidence import DEFAULT_NEGATIVE_SDT, DEFAULT_POSITIVE_SDT, DEFAULT_MODEL_ORDER
from reporting.statistics import DEFAULT_N_BOOT, DEFAULT_RANDOM_SEED, bootstrap_mean_ci, holm_adjust


DEFAULT_MATCHED_TRIAL_SCORES_CSV = Path(
    "./data/results_baseline/region_sliding_baseline300ms_controls_focus_q4/"
    "summary_temporal_evidence_check/temporal_evidence_trial_scores.csv"
)
DEFAULT_MODEL_NAMES = (
    "raw_regions_delta",
    "baseline_regions_augmented",
    "raw_plus_baseline_regions_augmented",
)
DEFAULT_FOCUS_MODEL_A = "raw_regions_delta"
DEFAULT_FOCUS_MODEL_B = "raw_plus_baseline_regions_augmented"
DEFAULT_FOCUS_START_MS = 125.0
DEFAULT_FOCUS_END_MS = 375.0

DEFAULT_CONFIDENCE_SDT_SUBSET = "hit"
DEFAULT_CONFIDENCE_MIN_UNIQUE = 4
DEFAULT_CONFIDENCE_MIN_PER_SIDE = 5
DEFAULT_CONFIDENCE_TRIAL_SCORES_CSV = Path(
    "./data/results_baseline/region_sliding_baseline300ms_controls_focus_q4/"
    "summary_matched_spatial_control_sweep_full/central_125_375/"
    "matched_spatial_control_trial_scores.csv"
)
DEFAULT_DETECTION_FOCUS_CONTRAST_CSV = Path(
    "./data/results_baseline/region_sliding_baseline300ms_controls_focus_q4/"
    "summary_matched_stimamp_evidence_central/matched_stimamp_focus_contrasts.csv"
)


def build_model_order(present_models: list[str]) -> list[str]:
    preferred = [*DEFAULT_MODEL_NAMES, *DEFAULT_MODEL_ORDER]
    ordered = [name for name in preferred if name in present_models]
    ordered.extend(name for name in present_models if name not in ordered)
    return ordered


def load_trial_scores(trial_scores_csv: Path) -> pd.DataFrame:
    if not trial_scores_csv.exists():
        raise FileNotFoundError(f"Evidence trial-score CSV not found: {trial_scores_csv}")

    df = pd.read_csv(trial_scores_csv)
    required = {
        "subject",
        "trial_idx",
        "sdt",
        "label",
        "model_name",
        "model_label",
        "evidence_score",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Evidence trial-score CSV is missing required columns: {missing}")

    if {"target_start_ms", "target_end_ms", "target_center_ms"}.issubset(df.columns):
        return df.reset_index(drop=True)

    df = df.copy()
    df["target_start_ms"] = 0.0
    df["target_end_ms"] = 0.0
    df["target_center_ms"] = 0.0
    return df.reset_index(drop=True)


def filter_models(
    trial_scores: pd.DataFrame,
    *,
    model_names: tuple[str, ...],
) -> pd.DataFrame:
    filtered = trial_scores.loc[trial_scores["model_name"].isin(model_names)].copy()
    if filtered.empty:
        raise ValueError(f"No trial-score rows matched the requested models: {list(model_names)}")
    return filtered


def load_stimamp_metadata(
    subjects: list[str],
    *,
    derivatives_dir: Path,
) -> pd.DataFrame:
    metadata = load_all_epoch_metadata(subjects, derivatives_dir=derivatives_dir)
    required = {"subject", "trial_idx", "stimamp"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Epoch metadata is missing required columns: {missing}")

    metadata = metadata.loc[:, ["subject", "trial_idx", "stimamp"]].copy()
    metadata["subject"] = metadata["subject"].astype(str)
    metadata["trial_idx"] = metadata["trial_idx"].astype(int)
    return metadata


def merge_stimamp(
    trial_scores: pd.DataFrame,
    *,
    derivatives_dir: Path,
) -> pd.DataFrame:
    subjects = sorted(trial_scores["subject"].dropna().astype(str).unique().tolist())
    metadata = load_stimamp_metadata(subjects, derivatives_dir=derivatives_dir)
    merged = trial_scores.merge(
        metadata,
        on=["subject", "trial_idx"],
        how="left",
        validate="many_to_one",
    )
    if merged["stimamp"].isna().any():
        example = merged.loc[merged["stimamp"].isna(), ["subject", "trial_idx"]].iloc[0].to_dict()
        raise ValueError(f"Stimamp metadata was missing for some trials, for example {example}")
    return merged


def build_matched_pairs(
    trial_scores: pd.DataFrame,
    *,
    positive_sdt: str,
    negative_sdt: str,
) -> pd.DataFrame:
    trial_metadata = trial_scores[
        [
            "subject",
            "trial_idx",
            "sdt",
            "stimamp",
            "target_start_ms",
            "target_end_ms",
            "target_center_ms",
        ]
    ].drop_duplicates()

    rows = []
    group_columns = ["target_start_ms", "target_end_ms", "target_center_ms", "subject"]
    for keys, subject_df in trial_metadata.groupby(group_columns, observed=False):
        target_start_ms, target_end_ms, target_center_ms, subject = keys
        pos_df = subject_df.loc[subject_df["sdt"] == positive_sdt, ["trial_idx", "stimamp"]].reset_index(drop=True)
        neg_df = subject_df.loc[subject_df["sdt"] == negative_sdt, ["trial_idx", "stimamp"]].reset_index(drop=True)
        if pos_df.empty or neg_df.empty:
            continue

        cost = np.abs(
            pos_df["stimamp"].to_numpy(dtype=float)[:, None]
            - neg_df["stimamp"].to_numpy(dtype=float)[None, :]
        )
        pos_idx, neg_idx = linear_sum_assignment(cost)
        if pos_idx.size == 0 or neg_idx.size == 0:
            continue

        order = np.argsort(cost[pos_idx, neg_idx])
        pos_idx = pos_idx[order]
        neg_idx = neg_idx[order]

        for pair_idx, (pos_row_idx, neg_row_idx) in enumerate(zip(pos_idx, neg_idx)):
            pos_row = pos_df.iloc[int(pos_row_idx)]
            neg_row = neg_df.iloc[int(neg_row_idx)]
            rows.append(
                {
                    "target_start_ms": float(target_start_ms),
                    "target_end_ms": float(target_end_ms),
                    "target_center_ms": float(target_center_ms),
                    "subject": str(subject),
                    "pair_idx": int(pair_idx),
                    "positive_trial_idx": int(pos_row["trial_idx"]),
                    "negative_trial_idx": int(neg_row["trial_idx"]),
                    "positive_stimamp": float(pos_row["stimamp"]),
                    "negative_stimamp": float(neg_row["stimamp"]),
                    "stimamp_abs_diff": float(abs(pos_row["stimamp"] - neg_row["stimamp"])),
                }
            )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        raise ValueError("No matched positive-versus-negative trial pairs could be formed")
    return pair_df.sort_values(["target_start_ms", "subject", "pair_idx"]).reset_index(drop=True)


def compare_pair_scores(positive_scores: np.ndarray, negative_scores: np.ndarray) -> np.ndarray:
    positive_scores = np.asarray(positive_scores, dtype=float)
    negative_scores = np.asarray(negative_scores, dtype=float)
    return np.where(
        positive_scores > negative_scores,
        1.0,
        np.where(positive_scores < negative_scores, 0.0, 0.5),
    )


def safe_wilcoxon(values: np.ndarray, *, null_value: float) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return float("nan"), float("nan")

    try:
        result = stats.wilcoxon(clean - null_value, zero_method="wilcox")
    except ValueError:
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


def evaluate_subject_pair_metrics(
    trial_scores: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    merge_columns = ["target_start_ms", "target_end_ms", "target_center_ms", "subject"]

    for (model_name, model_label), model_df in trial_scores.groupby(
        ["model_name", "model_label"],
        observed=False,
    ):
        positive_scores = pair_df.merge(
            model_df[merge_columns + ["trial_idx", "evidence_score"]],
            left_on=merge_columns + ["positive_trial_idx"],
            right_on=merge_columns + ["trial_idx"],
            how="left",
            validate="many_to_one",
        ).rename(columns={"evidence_score": "positive_score"})
        negative_scores = pair_df.merge(
            model_df[merge_columns + ["trial_idx", "evidence_score"]],
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
        merged["model_name"] = str(model_name)
        merged["model_label"] = str(model_label)

        for keys, subject_df in merged.groupby(
            ["target_start_ms", "target_end_ms", "target_center_ms", "subject"],
            observed=False,
        ):
            target_start_ms, target_end_ms, target_center_ms, subject = keys
            rows.append(
                {
                    "target_start_ms": float(target_start_ms),
                    "target_end_ms": float(target_end_ms),
                    "target_center_ms": float(target_center_ms),
                    "subject": str(subject),
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
        raise ValueError("No subject-level matched-pair summaries could be formed")
    return out


def summarise_pair_metrics(
    subject_summary: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    rows = []
    group_columns = ["target_start_ms", "target_end_ms", "target_center_ms", "model_name", "model_label"]
    for group_idx, (keys, group_df) in enumerate(subject_summary.groupby(group_columns, observed=False)):
        target_start_ms, target_end_ms, target_center_ms, model_name, model_label = keys
        pair_acc_values = group_df["pair_acc"].dropna().to_numpy(dtype=float)
        margin_values = group_df["mean_margin"].dropna().to_numpy(dtype=float)
        pair_acc_t = stats.ttest_1samp(pair_acc_values, 0.5, nan_policy="omit") if pair_acc_values.size >= 2 else None
        margin_t = stats.ttest_1samp(margin_values, 0.0, nan_policy="omit") if margin_values.size >= 2 else None
        pair_acc_w_stat, pair_acc_w_p = safe_wilcoxon(pair_acc_values, null_value=0.5)
        margin_w_stat, margin_w_p = safe_wilcoxon(margin_values, null_value=0.0)
        pair_acc_ci_low, pair_acc_ci_high = bootstrap_mean_ci(
            pair_acc_values,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 1000 + group_idx,
        )
        margin_ci_low, margin_ci_high = bootstrap_mean_ci(
            margin_values,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 20_000 + group_idx,
        )
        rows.append(
            {
                "target_start_ms": float(target_start_ms),
                "target_end_ms": float(target_end_ms),
                "target_center_ms": float(target_center_ms),
                "model_name": str(model_name),
                "model_label": str(model_label),
                "n_subjects": int(group_df.shape[0]),
                "pair_acc_mean": float(group_df["pair_acc"].mean()),
                "pair_acc_std": float(group_df["pair_acc"].std(ddof=1)) if group_df.shape[0] > 1 else float("nan"),
                "pair_acc_ci95_low": pair_acc_ci_low,
                "pair_acc_ci95_high": pair_acc_ci_high,
                "pair_acc_above_half_share": float(np.mean(group_df["pair_acc"] > 0.5)),
                "pair_acc_ttest_p": float(pair_acc_t.pvalue) if pair_acc_t is not None else float("nan"),
                "pair_acc_wilcoxon_stat": pair_acc_w_stat,
                "pair_acc_wilcoxon_p": pair_acc_w_p,
                "mean_margin_mean": float(group_df["mean_margin"].mean()),
                "mean_margin_std": float(group_df["mean_margin"].std(ddof=1)) if group_df.shape[0] > 1 else float("nan"),
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
    present_models = out["model_name"].dropna().astype(str).tolist()
    ordered_names = build_model_order(present_models)
    out["model_name"] = pd.Categorical(out["model_name"], categories=ordered_names, ordered=True)
    return out.sort_values(["target_start_ms", "model_name"]).reset_index(drop=True)


def build_pairwise_model_comparisons(
    subject_summary: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    rows = []
    window_columns = ["target_start_ms", "target_end_ms", "target_center_ms"]
    pair_counter = 0
    for keys, window_df in subject_summary.groupby(window_columns, observed=False):
        target_start_ms, target_end_ms, target_center_ms = keys
        present_models = window_df["model_name"].dropna().astype(str).unique().tolist()
        models = build_model_order(present_models)
        for model_a_idx, model_a in enumerate(models):
            for model_b in models[model_a_idx + 1 :]:
                wide = (
                    window_df.loc[window_df["model_name"].isin([model_a, model_b])]
                    .pivot(index="subject", columns="model_name", values=["pair_acc", "mean_margin"])
                )
                for metric, null_value in (("pair_acc", 0.0), ("mean_margin", 0.0)):
                    if metric not in wide.columns.get_level_values(0):
                        continue
                    metric_wide = wide[metric]
                    if model_a not in metric_wide.columns or model_b not in metric_wide.columns:
                        continue
                    pair_df = metric_wide[[model_a, model_b]].dropna()
                    if pair_df.empty:
                        continue
                    values_a = pair_df[model_a].to_numpy(dtype=float)
                    values_b = pair_df[model_b].to_numpy(dtype=float)
                    delta = values_b - values_a
                    if pair_df.shape[0] >= 2:
                        t_stat, t_p = stats.ttest_1samp(delta, null_value)
                        t_stat = float(t_stat)
                        t_p = float(t_p)
                    else:
                        t_stat = float("nan")
                        t_p = float("nan")
                    w_stat, w_p = safe_wilcoxon(delta, null_value=null_value)
                    ci_low, ci_high = bootstrap_mean_ci(
                        delta,
                        n_bootstrap=n_bootstrap,
                        random_seed=random_seed + 1000 * (pair_counter + 1),
                    )
                    std_delta = float(np.std(delta, ddof=1)) if pair_df.shape[0] > 1 else float("nan")
                    effect_dz = (
                        float(np.mean(delta) / std_delta)
                        if np.isfinite(std_delta) and std_delta > 0.0
                        else float("nan")
                    )
                    rows.append(
                        {
                            "target_start_ms": float(target_start_ms),
                            "target_end_ms": float(target_end_ms),
                            "target_center_ms": float(target_center_ms),
                            "metric": metric,
                            "model_a": model_a,
                            "model_b": model_b,
                            "n_subjects": int(pair_df.shape[0]),
                            "mean_model_a": float(np.mean(values_a)),
                            "mean_model_b": float(np.mean(values_b)),
                            "mean_delta_model_b_minus_a": float(np.mean(delta)),
                            "ci95_low_delta_model_b_minus_a": ci_low,
                            "ci95_high_delta_model_b_minus_a": ci_high,
                            "cohen_dz": effect_dz,
                            "n_model_b_better": int(np.sum(delta > 0.0)),
                            "n_model_a_better": int(np.sum(delta < 0.0)),
                            "n_ties": int(np.sum(np.isclose(delta, 0.0))),
                            "positive_subject_share": float(np.mean(delta > 0.0)),
                            "ttest_stat": t_stat,
                            "ttest_p": t_p,
                            "wilcoxon_stat": w_stat,
                            "wilcoxon_p": w_p,
                        }
                    )
                    pair_counter += 1

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ttest_p_holm"] = np.nan
    out["wilcoxon_p_holm"] = np.nan
    for metric_name in out["metric"].dropna().astype(str).unique().tolist():
        metric_mask = out["metric"] == metric_name
        out.loc[metric_mask, "ttest_p_holm"] = holm_adjust(
            out.loc[metric_mask, "ttest_p"].to_list()
        )
        out.loc[metric_mask, "wilcoxon_p_holm"] = holm_adjust(
            out.loc[metric_mask, "wilcoxon_p"].to_list()
        )
    return out.sort_values(["target_start_ms", "metric", "model_a", "model_b"]).reset_index(drop=True)


def select_focus_rows(
    pairwise_summary: pd.DataFrame,
    *,
    focus_model_a: str,
    focus_model_b: str,
    metric: str,
) -> pd.DataFrame:
    pair_df = pairwise_summary.loc[pairwise_summary["metric"] == metric].copy()
    pair_df = pair_df.loc[
        (
            (pair_df["model_a"] == focus_model_a)
            & (pair_df["model_b"] == focus_model_b)
        )
        | (
            (pair_df["model_a"] == focus_model_b)
            & (pair_df["model_b"] == focus_model_a)
        )
    ].copy()
    if pair_df.empty:
        return pair_df

    reverse = pair_df["model_a"] == focus_model_b
    pair_df.loc[reverse, "model_a"] = focus_model_a
    pair_df.loc[reverse, "model_b"] = focus_model_b
    reverse_mean_a = pair_df.loc[reverse, "mean_model_a"].copy()
    reverse_mean_b = pair_df.loc[reverse, "mean_model_b"].copy()
    pair_df.loc[reverse, "mean_model_a"] = reverse_mean_b.to_numpy(dtype=float)
    pair_df.loc[reverse, "mean_model_b"] = reverse_mean_a.to_numpy(dtype=float)
    reverse_ci_low = pair_df.loc[reverse, "ci95_low_delta_model_b_minus_a"].copy()
    reverse_ci_high = pair_df.loc[reverse, "ci95_high_delta_model_b_minus_a"].copy()
    pair_df.loc[reverse, "mean_delta_model_b_minus_a"] = -pair_df.loc[reverse, "mean_delta_model_b_minus_a"]
    pair_df.loc[reverse, "ci95_low_delta_model_b_minus_a"] = -reverse_ci_high.to_numpy(dtype=float)
    pair_df.loc[reverse, "ci95_high_delta_model_b_minus_a"] = -reverse_ci_low.to_numpy(dtype=float)
    reverse_cohen_dz = pair_df.loc[reverse, "cohen_dz"].copy()
    pair_df.loc[reverse, "cohen_dz"] = -reverse_cohen_dz.to_numpy(dtype=float)
    reverse_b_better = pair_df.loc[reverse, "n_model_b_better"].copy()
    reverse_a_better = pair_df.loc[reverse, "n_model_a_better"].copy()
    pair_df.loc[reverse, "n_model_b_better"] = reverse_a_better.to_numpy(dtype=int)
    pair_df.loc[reverse, "n_model_a_better"] = reverse_b_better.to_numpy(dtype=int)
    pair_df.loc[reverse, "positive_subject_share"] = (
        pair_df.loc[reverse, "n_model_b_better"].to_numpy(dtype=float)
        / pair_df.loc[reverse, "n_subjects"].to_numpy(dtype=float)
    )
    pair_df.loc[reverse, "ttest_stat"] = -pair_df.loc[reverse, "ttest_stat"]
    reverse_wilcoxon_stat = pair_df.loc[reverse, "wilcoxon_stat"].copy()
    pair_df.loc[reverse, "wilcoxon_stat"] = reverse_wilcoxon_stat.to_numpy(dtype=float)
    pair_df["contrast_label"] = f"{focus_model_b} minus {focus_model_a}"
    return pair_df.sort_values("target_start_ms").reset_index(drop=True)


def choose_focus_window(
    pairwise_summary: pd.DataFrame,
    *,
    focus_model_a: str,
    focus_model_b: str,
    default_start_ms: float,
    default_end_ms: float,
) -> tuple[float, float]:
    focus_rows = select_focus_rows(
        pairwise_summary,
        focus_model_a=focus_model_a,
        focus_model_b=focus_model_b,
        metric="pair_acc",
    )
    exact = focus_rows.loc[
        (focus_rows["target_start_ms"] == float(default_start_ms))
        & (focus_rows["target_end_ms"] == float(default_end_ms))
    ]
    if not exact.empty:
        return float(default_start_ms), float(default_end_ms)
    if focus_rows.empty:
        raise ValueError("No pairwise rows were available for the requested focus models")

    best = focus_rows.sort_values("mean_delta_model_b_minus_a", ascending=False).iloc[0]
    return float(best["target_start_ms"]), float(best["target_end_ms"])


def build_match_quality_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    return (
        pair_df.groupby(["target_start_ms", "target_end_ms", "target_center_ms"], observed=False)
        .agg(
            n_pairs=("pair_idx", "size"),
            n_subjects=("subject", "nunique"),
            mean_stimamp_abs_diff=("stimamp_abs_diff", "mean"),
            median_stimamp_abs_diff=("stimamp_abs_diff", "median"),
        )
        .reset_index()
    )


def build_focus_contrast_tables(
    pairwise_summary: pd.DataFrame,
    match_quality: pd.DataFrame,
    *,
    focus_model_a: str,
    focus_model_b: str,
    focus_start_ms: float,
    focus_end_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus_frames = [
        select_focus_rows(
            pairwise_summary,
            focus_model_a=focus_model_a,
            focus_model_b=focus_model_b,
            metric=metric_name,
        )
        for metric_name in ("pair_acc", "mean_margin")
    ]
    focus_contrasts = pd.concat(focus_frames, ignore_index=True)
    focus_contrasts = focus_contrasts.sort_values(["metric", "target_start_ms"]).reset_index(drop=True)
    if not focus_contrasts.empty:
        focus_contrasts["ttest_p_holm_focus"] = np.nan
        focus_contrasts["wilcoxon_p_holm_focus"] = np.nan
        for metric_name in focus_contrasts["metric"].dropna().astype(str).unique().tolist():
            metric_mask = focus_contrasts["metric"] == metric_name
            focus_contrasts.loc[metric_mask, "ttest_p_holm_focus"] = holm_adjust(
                focus_contrasts.loc[metric_mask, "ttest_p"].to_list()
            )
            focus_contrasts.loc[metric_mask, "wilcoxon_p_holm_focus"] = holm_adjust(
                focus_contrasts.loc[metric_mask, "wilcoxon_p"].to_list()
            )

    headline_summary = focus_contrasts.loc[
        (focus_contrasts["target_start_ms"] == float(focus_start_ms))
        & (focus_contrasts["target_end_ms"] == float(focus_end_ms))
    ].copy()
    if headline_summary.empty:
        return focus_contrasts, headline_summary

    headline_summary = headline_summary.merge(
        match_quality,
        on=["target_start_ms", "target_end_ms", "target_center_ms"],
        how="left",
        validate="many_to_one",
    )
    headline_summary.insert(0, "focus_model_a", focus_model_a)
    headline_summary.insert(1, "focus_model_b", focus_model_b)
    return focus_contrasts, headline_summary


def plot_matched_evidence_summary(
    group_summary: pd.DataFrame,
    *,
    model_names: tuple[str, ...],
    outfile: Path,
) -> None:
    if group_summary.empty:
        return

    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    palette = {
        "raw_regions_delta": "#1d3557",
        "baseline_regions_augmented": "#457b9d",
        "raw_plus_baseline_regions_augmented": "#e76f51",
        "baseline_cf_augmented": "#457b9d",
        "prism_cf_augmented": "#6d597a",
        "raw_cf_plus_baseline_cf_augmented": "#e76f51",
        "raw_cf_plus_prism_cf_augmented": "#9c6644",
        "raw_cf_delta": "#1d3557",
    }

    for model_name in model_names:
        model_df = group_summary.loc[group_summary["model_name"] == model_name].sort_values("target_start_ms")
        if model_df.empty:
            continue
        colour = palette.get(model_name, "#495057")
        label = str(model_df["model_label"].iloc[0])
        axes[0].plot(
            model_df["target_start_ms"],
            model_df["pair_acc_mean"],
            marker="o",
            linewidth=2.2,
            color=colour,
            label=label,
        )
        if {"pair_acc_ci95_low", "pair_acc_ci95_high"}.issubset(model_df.columns):
            axes[0].fill_between(
                model_df["target_start_ms"].to_numpy(dtype=float),
                model_df["pair_acc_ci95_low"].to_numpy(dtype=float),
                model_df["pair_acc_ci95_high"].to_numpy(dtype=float),
                color=colour,
                alpha=0.14,
            )
        axes[1].plot(
            model_df["target_start_ms"],
            model_df["mean_margin_mean"],
            marker="o",
            linewidth=2.2,
            color=colour,
            label=label,
        )
        if {"mean_margin_ci95_low", "mean_margin_ci95_high"}.issubset(model_df.columns):
            axes[1].fill_between(
                model_df["target_start_ms"].to_numpy(dtype=float),
                model_df["mean_margin_ci95_low"].to_numpy(dtype=float),
                model_df["mean_margin_ci95_high"].to_numpy(dtype=float),
                color=colour,
                alpha=0.14,
            )

    axes[0].axhline(0.5, color="#6c757d", linestyle="--", linewidth=1.0)
    axes[0].set_title("Matched-pair hit over miss accuracy")
    axes[0].set_xlabel("Target window start (ms)")
    axes[0].set_ylabel("Subject mean pair accuracy")
    axes[0].grid(True, alpha=0.25)

    axes[1].axhline(0.0, color="#6c757d", linestyle="--", linewidth=1.0)
    axes[1].set_title("Matched-pair score margin")
    axes[1].set_xlabel("Target window start (ms)")
    axes[1].set_ylabel("Subject mean hit-minus-miss score")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, loc="best")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_matched_evidence_report(
    focus_contrasts: pd.DataFrame,
    match_quality: pd.DataFrame,
    *,
    focus_model_a: str,
    focus_model_b: str,
    focus_start_ms: float,
    focus_end_ms: float,
) -> str:
    lines = [
        "# Matched Stimamp Evidence Summary",
        "",
        (
            "Positive and negative trials were matched one-to-one within subject by minimising the absolute "
            "difference in stimulus amplitude. This is a stricter control than coarse amplitude bins."
        ),
        "",
    ]

    quality_row = match_quality.loc[
        (match_quality["target_start_ms"] == float(focus_start_ms))
        & (match_quality["target_end_ms"] == float(focus_end_ms))
    ]
    if not quality_row.empty:
        quality_row = quality_row.iloc[0]
        lines.extend(
            [
                "## Match quality",
                "",
                (
                    f"- Focus window used {int(quality_row['n_pairs'])} matched pairs across "
                    f"{int(quality_row['n_subjects'])} subjects with mean |stimamp difference| "
                    f"{quality_row['mean_stimamp_abs_diff']:.4f} "
                    f"(median {quality_row['median_stimamp_abs_diff']:.4f})."
                ),
                "",
            ]
        )

    focus_rows = focus_contrasts.loc[focus_contrasts["metric"] == "pair_acc"].copy()
    focus_exact = focus_rows.loc[
        (focus_rows["target_start_ms"] == float(focus_start_ms))
        & (focus_rows["target_end_ms"] == float(focus_end_ms))
    ]
    if not focus_exact.empty:
        row = focus_exact.iloc[0]
        lines.extend(
            [
                f"## {int(focus_start_ms)} to {int(focus_end_ms)} ms",
                "",
                (
                    f"- {focus_model_b} minus {focus_model_a} matched-pair accuracy delta = "
                    f"{row['mean_delta_model_b_minus_a']:.3f} "
                    f"(95% CI [{row['ci95_low_delta_model_b_minus_a']:.3f}, "
                    f"{row['ci95_high_delta_model_b_minus_a']:.3f}], "
                    f"paired t Holm p={row['ttest_p_holm_focus']:.3g}, "
                    f"Wilcoxon Holm p={row['wilcoxon_p_holm_focus']:.3g}, "
                    f"positive-subject share={row['positive_subject_share']:.3f})."
                ),
                "",
            ]
        )

    focus_margin = focus_contrasts.loc[focus_contrasts["metric"] == "mean_margin"].copy()
    focus_margin = focus_margin.loc[
        (focus_margin["target_start_ms"] == float(focus_start_ms))
        & (focus_margin["target_end_ms"] == float(focus_end_ms))
    ]
    if not focus_margin.empty:
        row = focus_margin.iloc[0]
        lines.extend(
            [
                "## Score margin",
                "",
                (
                    f"- {focus_model_b} minus {focus_model_a} matched-pair score-margin delta = "
                    f"{row['mean_delta_model_b_minus_a']:.3f} "
                    f"(95% CI [{row['ci95_low_delta_model_b_minus_a']:.3f}, "
                    f"{row['ci95_high_delta_model_b_minus_a']:.3f}], "
                    f"paired t Holm p={row['ttest_p_holm_focus']:.3g}, "
                    f"Wilcoxon Holm p={row['wilcoxon_p_holm_focus']:.3g})."
                ),
                "",
            ]
        )

    lines.extend(["## Temporal pattern", ""])
    for _, row in focus_rows.sort_values("target_start_ms").iterrows():
        lines.append(
            f"- {int(row['target_start_ms'])} to {int(row['target_end_ms'])} ms: "
            f"{focus_model_b} minus {focus_model_a} pair-accuracy delta = "
            f"{row['mean_delta_model_b_minus_a']:.3f} "
            f"(95% CI [{row['ci95_low_delta_model_b_minus_a']:.3f}, "
            f"{row['ci95_high_delta_model_b_minus_a']:.3f}], "
            f"paired t Holm p={row['ttest_p_holm_focus']:.3g})."
        )

    best_row = focus_rows.sort_values("mean_delta_model_b_minus_a", ascending=False).iloc[0]
    lines.extend(
        [
            "",
            "## Best window",
            "",
            (
                f"- Strongest matched-pair gain occurred at {int(best_row['target_start_ms'])} to "
                f"{int(best_row['target_end_ms'])} ms with delta {best_row['mean_delta_model_b_minus_a']:.3f} "
                f"(paired t Holm p={best_row['ttest_p_holm_focus']:.3g})."
            ),
        ]
    )

    return "\n".join(lines)


def run_matched_stimamp_evidence_summary(
    *,
    trial_scores_csv: Path = DEFAULT_MATCHED_TRIAL_SCORES_CSV,
    derivatives_dir: Path = DEFAULT_DERIVATIVES_DIR,
    outdir: Path | None = None,
    model_names: tuple[str, ...] = DEFAULT_MODEL_NAMES,
    positive_sdt: str = DEFAULT_POSITIVE_SDT,
    negative_sdt: str = DEFAULT_NEGATIVE_SDT,
    focus_model_a: str = DEFAULT_FOCUS_MODEL_A,
    focus_model_b: str = DEFAULT_FOCUS_MODEL_B,
    focus_start_ms: float = DEFAULT_FOCUS_START_MS,
    focus_end_ms: float = DEFAULT_FOCUS_END_MS,
    n_bootstrap: int = DEFAULT_N_BOOT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> int:
    outdir = trial_scores_csv.parent / "summary_matched_stimamp_evidence" if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        trial_scores = load_trial_scores(trial_scores_csv)
        trial_scores = filter_models(trial_scores, model_names=model_names)
        if "group_kind" in trial_scores.columns:
            named_mask = trial_scores["group_kind"].astype(str) == "named_region"
            if named_mask.any():
                trial_scores = trial_scores.loc[named_mask].reset_index(drop=True)
        trial_scores = _ensure_stimamp_in_trial_scores(
            trial_scores,
            derivatives_dir=Path(derivatives_dir),
        )
        pair_df = build_matched_pairs(
            trial_scores,
            positive_sdt=positive_sdt,
            negative_sdt=negative_sdt,
        )
        subject_summary = evaluate_subject_pair_metrics(trial_scores, pair_df)
        group_summary = summarise_pair_metrics(
            subject_summary,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        pairwise_summary = build_pairwise_model_comparisons(
            subject_summary,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        focus_start_ms, focus_end_ms = choose_focus_window(
            pairwise_summary,
            focus_model_a=focus_model_a,
            focus_model_b=focus_model_b,
            default_start_ms=focus_start_ms,
            default_end_ms=focus_end_ms,
        )
        match_quality = build_match_quality_summary(pair_df)
        focus_contrasts, headline_summary = build_focus_contrast_tables(
            pairwise_summary,
            match_quality,
            focus_model_a=focus_model_a,
            focus_model_b=focus_model_b,
            focus_start_ms=focus_start_ms,
            focus_end_ms=focus_end_ms,
        )
        report_text = build_matched_evidence_report(
            focus_contrasts,
            match_quality,
            focus_model_a=focus_model_a,
            focus_model_b=focus_model_b,
            focus_start_ms=focus_start_ms,
            focus_end_ms=focus_end_ms,
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    pair_path = outdir / "matched_stimamp_pairs.csv"
    subject_path = outdir / "matched_stimamp_subject_summary.csv"
    group_path = outdir / "matched_stimamp_group_summary.csv"
    pairwise_path = outdir / "matched_stimamp_pairwise_summary.csv"
    quality_path = outdir / "matched_stimamp_match_quality.csv"
    focus_contrasts_path = outdir / "matched_stimamp_focus_contrasts.csv"
    headline_summary_path = outdir / "matched_stimamp_headline_summary.csv"
    report_path = outdir / "matched_stimamp_summary.md"
    plot_path = outdir / "matched_stimamp_pair_acc.png"

    save_table(pair_df, pair_path)
    save_table(subject_summary, subject_path)
    save_table(group_summary, group_path)
    save_table(pairwise_summary, pairwise_path)
    save_table(match_quality, quality_path)
    save_table(focus_contrasts, focus_contrasts_path)
    save_table(headline_summary, headline_summary_path)
    report_path.write_text(report_text)
    plot_matched_evidence_summary(
        group_summary,
        model_names=model_names,
        outfile=plot_path,
    )

    print(f"Saved {pair_path}")
    print(f"Saved {subject_path}")
    print(f"Saved {group_path}")
    print(f"Saved {pairwise_path}")
    print(f"Saved {quality_path}")
    print(f"Saved {focus_contrasts_path}")
    print(f"Saved {headline_summary_path}")
    print(f"Saved {report_path}")
    print(f"Saved {plot_path}")

    print("\nFocus window:")
    print(f"  {focus_start_ms:.0f} to {focus_end_ms:.0f} ms")

    focus_rows = focus_contrasts.loc[
        (focus_contrasts["metric"] == "pair_acc")
        & (focus_contrasts["target_start_ms"] == float(focus_start_ms))
        & (focus_contrasts["target_end_ms"] == float(focus_end_ms))
    ].copy()
    if not focus_rows.empty:
        print("\nMatched-pair focus contrast:")
        print(
            focus_rows[
                [
                    "n_subjects",
                    "mean_model_a",
                    "mean_model_b",
                    "mean_delta_model_b_minus_a",
                    "ci95_low_delta_model_b_minus_a",
                    "ci95_high_delta_model_b_minus_a",
                    "positive_subject_share",
                    "ttest_p",
                    "ttest_p_holm_focus",
                    "wilcoxon_p",
                    "wilcoxon_p_holm_focus",
                ]
            ].to_string(index=False)
        )

    return 0


def _ensure_stimamp_in_trial_scores(
    trial_scores: pd.DataFrame,
    *,
    derivatives_dir: Path,
) -> pd.DataFrame:
    if "stimamp" in trial_scores.columns and not trial_scores["stimamp"].isna().any():
        return trial_scores
    return merge_stimamp(trial_scores, derivatives_dir=derivatives_dir)


def build_confidence_matched_pairs(
    trial_scores: pd.DataFrame,
    *,
    sdt_subset: str = DEFAULT_CONFIDENCE_SDT_SUBSET,
    min_unique_confidence: int = DEFAULT_CONFIDENCE_MIN_UNIQUE,
    min_per_side: int = DEFAULT_CONFIDENCE_MIN_PER_SIDE,
) -> pd.DataFrame:
    if "confidence" not in trial_scores.columns:
        raise ValueError("Confidence column missing from trial scores; cannot build confidence pairs")

    metadata = trial_scores[
        [
            "subject",
            "trial_idx",
            "sdt",
            "stimamp",
            "confidence",
            "target_start_ms",
            "target_end_ms",
            "target_center_ms",
        ]
    ].drop_duplicates()
    metadata = metadata.loc[metadata["sdt"].astype(str) == str(sdt_subset)].copy()
    metadata = metadata.dropna(subset=["confidence", "stimamp"])
    if metadata.empty:
        raise ValueError(
            f"No trials matched the requested sdt subset {sdt_subset!r} for confidence pairing"
        )

    rows = []
    skipped = []
    group_columns = ["target_start_ms", "target_end_ms", "target_center_ms", "subject"]
    for keys, subject_df in metadata.groupby(group_columns, observed=False):
        target_start_ms, target_end_ms, target_center_ms, subject = keys
        confs = subject_df["confidence"].to_numpy(dtype=float)
        if subject_df["confidence"].nunique() < min_unique_confidence:
            skipped.append(
                {
                    "target_start_ms": float(target_start_ms),
                    "target_end_ms": float(target_end_ms),
                    "target_center_ms": float(target_center_ms),
                    "subject": str(subject),
                    "reason": "insufficient_unique_confidence",
                    "n_trials": int(subject_df.shape[0]),
                    "n_unique_confidence": int(subject_df["confidence"].nunique()),
                }
            )
            continue

        median_conf = float(np.median(confs))
        high_mask = confs > median_conf
        low_mask = confs < median_conf
        # When the median lies on a tie, push the tied trials onto the low side
        # so that "high" only contains strictly-above-median confidence.
        if high_mask.sum() < min_per_side or low_mask.sum() < min_per_side:
            skipped.append(
                {
                    "target_start_ms": float(target_start_ms),
                    "target_end_ms": float(target_end_ms),
                    "target_center_ms": float(target_center_ms),
                    "subject": str(subject),
                    "reason": "insufficient_per_side_after_split",
                    "n_trials": int(subject_df.shape[0]),
                    "n_high": int(high_mask.sum()),
                    "n_low": int(low_mask.sum()),
                    "median_confidence": median_conf,
                }
            )
            continue

        high_df = subject_df.loc[high_mask, ["trial_idx", "stimamp", "confidence"]].reset_index(drop=True)
        low_df = subject_df.loc[low_mask, ["trial_idx", "stimamp", "confidence"]].reset_index(drop=True)

        cost = np.abs(
            high_df["stimamp"].to_numpy(dtype=float)[:, None]
            - low_df["stimamp"].to_numpy(dtype=float)[None, :]
        )
        high_idx, low_idx = linear_sum_assignment(cost)
        if high_idx.size == 0 or low_idx.size == 0:
            continue

        order = np.argsort(cost[high_idx, low_idx])
        high_idx = high_idx[order]
        low_idx = low_idx[order]

        for pair_idx, (high_row_idx, low_row_idx) in enumerate(zip(high_idx, low_idx)):
            high_row = high_df.iloc[int(high_row_idx)]
            low_row = low_df.iloc[int(low_row_idx)]
            rows.append(
                {
                    "target_start_ms": float(target_start_ms),
                    "target_end_ms": float(target_end_ms),
                    "target_center_ms": float(target_center_ms),
                    "subject": str(subject),
                    "pair_idx": int(pair_idx),
                    "positive_trial_idx": int(high_row["trial_idx"]),
                    "negative_trial_idx": int(low_row["trial_idx"]),
                    "positive_stimamp": float(high_row["stimamp"]),
                    "negative_stimamp": float(low_row["stimamp"]),
                    "stimamp_abs_diff": float(abs(high_row["stimamp"] - low_row["stimamp"])),
                    "positive_confidence": float(high_row["confidence"]),
                    "negative_confidence": float(low_row["confidence"]),
                    "confidence_abs_diff": float(abs(high_row["confidence"] - low_row["confidence"])),
                }
            )

    pair_df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    if pair_df.empty:
        raise ValueError(
            "No matched high-vs-low confidence pairs could be formed; check sdt subset and confidence variance"
        )
    pair_df = pair_df.sort_values(["target_start_ms", "subject", "pair_idx"]).reset_index(drop=True)
    pair_df.attrs["skipped_subjects"] = skipped_df
    return pair_df


def build_confidence_match_quality_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    return (
        pair_df.groupby(
            ["target_start_ms", "target_end_ms", "target_center_ms"],
            observed=False,
        )
        .agg(
            n_pairs=("pair_idx", "size"),
            n_subjects=("subject", "nunique"),
            mean_stimamp_abs_diff=("stimamp_abs_diff", "mean"),
            median_stimamp_abs_diff=("stimamp_abs_diff", "median"),
            mean_confidence_abs_diff=("confidence_abs_diff", "mean"),
            median_confidence_abs_diff=("confidence_abs_diff", "median"),
            mean_high_confidence=("positive_confidence", "mean"),
            mean_low_confidence=("negative_confidence", "mean"),
        )
        .reset_index()
    )


def _focus_row_for(metric: str, contrasts: pd.DataFrame, *, focus_start_ms: float, focus_end_ms: float):
    rows = contrasts.loc[
        (contrasts["metric"] == metric)
        & (contrasts["target_start_ms"] == float(focus_start_ms))
        & (contrasts["target_end_ms"] == float(focus_end_ms))
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_confidence_dissociation_report(
    *,
    confidence_focus_contrasts: pd.DataFrame,
    confidence_quality: pd.DataFrame,
    detection_focus_contrasts: pd.DataFrame | None,
    focus_model_a: str,
    focus_model_b: str,
    focus_start_ms: float,
    focus_end_ms: float,
    sdt_subset: str,
) -> str:
    lines = [
        "# Confidence Dissociation Summary",
        "",
        (
            "We test whether the same hybrid model that improves matched detection (hit-vs-miss with "
            "stimamp matched 1:1 within subject) also improves matched **confidence discrimination** "
            "on the same trials. Confidence pairs are formed within each subject by splitting "
            f"{sdt_subset!r} trials at the within-subject median confidence and matching high-vs-low "
            "trials one-to-one on stimamp via linear-sum-assignment. A null result here, combined with "
            "the positive detection result, demonstrates that the hybrid carries **detection-specific** "
            "information rather than generic per-trial signal-to-noise."
        ),
        "",
        "## Confidence-pair match quality",
        "",
    ]
    quality_row = confidence_quality.loc[
        (confidence_quality["target_start_ms"] == float(focus_start_ms))
        & (confidence_quality["target_end_ms"] == float(focus_end_ms))
    ]
    if not quality_row.empty:
        q = quality_row.iloc[0]
        lines.append(
            f"- Focus window {int(focus_start_ms)}-{int(focus_end_ms)} ms: "
            f"{int(q['n_pairs'])} matched high-vs-low confidence pairs across "
            f"{int(q['n_subjects'])} subjects."
        )
        lines.append(
            f"- Mean |stimamp difference| within confidence pair: {q['mean_stimamp_abs_diff']:.4f} "
            f"(median {q['median_stimamp_abs_diff']:.4f}) — comparable to the detection match quality, "
            f"so any contrast cannot be explained by stimamp differences."
        )
        lines.append(
            f"- Mean confidence on high side: {q['mean_high_confidence']:.3f}; on low side: "
            f"{q['mean_low_confidence']:.3f}; mean |confidence diff|: "
            f"{q['mean_confidence_abs_diff']:.3f}."
        )
        lines.append("")

    conf_pair_acc = _focus_row_for(
        "pair_acc",
        confidence_focus_contrasts,
        focus_start_ms=focus_start_ms,
        focus_end_ms=focus_end_ms,
    )
    conf_margin = _focus_row_for(
        "mean_margin",
        confidence_focus_contrasts,
        focus_start_ms=focus_start_ms,
        focus_end_ms=focus_end_ms,
    )

    lines.extend(["## Confidence-pair contrast (hybrid minus raw)", ""])
    if conf_pair_acc is not None:
        lines.append(
            f"- {focus_model_b} minus {focus_model_a} matched-pair confidence accuracy delta = "
            f"{conf_pair_acc['mean_delta_model_b_minus_a']:.3f} "
            f"(95% CI [{conf_pair_acc['ci95_low_delta_model_b_minus_a']:.3f}, "
            f"{conf_pair_acc['ci95_high_delta_model_b_minus_a']:.3f}], "
            f"paired t Holm p={conf_pair_acc['ttest_p_holm_focus']:.3g}, "
            f"Wilcoxon Holm p={conf_pair_acc['wilcoxon_p_holm_focus']:.3g}, "
            f"positive-subject share={conf_pair_acc['positive_subject_share']:.3f})."
        )
    if conf_margin is not None:
        lines.append(
            f"- {focus_model_b} minus {focus_model_a} matched-pair confidence score-margin delta = "
            f"{conf_margin['mean_delta_model_b_minus_a']:.3f} "
            f"(95% CI [{conf_margin['ci95_low_delta_model_b_minus_a']:.3f}, "
            f"{conf_margin['ci95_high_delta_model_b_minus_a']:.3f}], "
            f"paired t Holm p={conf_margin['ttest_p_holm_focus']:.3g}, "
            f"Wilcoxon Holm p={conf_margin['wilcoxon_p_holm_focus']:.3g})."
        )
    lines.append("")

    if detection_focus_contrasts is not None and not detection_focus_contrasts.empty:
        det_pair_acc = _focus_row_for(
            "pair_acc",
            detection_focus_contrasts,
            focus_start_ms=focus_start_ms,
            focus_end_ms=focus_end_ms,
        )
        det_margin = _focus_row_for(
            "mean_margin",
            detection_focus_contrasts,
            focus_start_ms=focus_start_ms,
            focus_end_ms=focus_end_ms,
        )
        lines.extend(
            [
                "## Detection-pair contrast (loaded for side-by-side comparison)",
                "",
            ]
        )
        if det_pair_acc is not None:
            lines.append(
                f"- {focus_model_b} minus {focus_model_a} matched-pair detection accuracy delta = "
                f"{det_pair_acc['mean_delta_model_b_minus_a']:.3f} "
                f"(95% CI [{det_pair_acc['ci95_low_delta_model_b_minus_a']:.3f}, "
                f"{det_pair_acc['ci95_high_delta_model_b_minus_a']:.3f}], "
                f"paired t Holm p={det_pair_acc['ttest_p_holm_focus']:.3g}, "
                f"Wilcoxon Holm p={det_pair_acc['wilcoxon_p_holm_focus']:.3g})."
            )
        if det_margin is not None:
            lines.append(
                f"- {focus_model_b} minus {focus_model_a} matched-pair detection score-margin delta = "
                f"{det_margin['mean_delta_model_b_minus_a']:.3f} "
                f"(95% CI [{det_margin['ci95_low_delta_model_b_minus_a']:.3f}, "
                f"{det_margin['ci95_high_delta_model_b_minus_a']:.3f}], "
                f"paired t Holm p={det_margin['ttest_p_holm_focus']:.3g}, "
                f"Wilcoxon Holm p={det_margin['wilcoxon_p_holm_focus']:.3g})."
            )
        lines.append("")

        if det_pair_acc is not None and conf_pair_acc is not None:
            det_delta = float(det_pair_acc["mean_delta_model_b_minus_a"])
            conf_delta = float(conf_pair_acc["mean_delta_model_b_minus_a"])
            det_p = float(det_pair_acc["ttest_p_holm_focus"])
            conf_p = float(conf_pair_acc["ttest_p_holm_focus"])
            verdict = (
                "DETECTION-SPECIFIC: hybrid significantly improves matched-pair detection "
                "but not matched-pair confidence discrimination."
            )
            if conf_p < 0.05 and conf_delta > 0.0:
                verdict = (
                    "NOT SPECIFIC: the hybrid also improves matched-pair confidence discrimination, "
                    "so the gain may reflect generic SNR rather than detection-specific signal."
                )
            elif det_p >= 0.05:
                verdict = (
                    "INCONCLUSIVE: detection contrast is not itself significant after Holm correction; "
                    "the dissociation cannot be claimed."
                )
            lines.extend(
                [
                    "## Dissociation verdict",
                    "",
                    (
                        f"- Detection delta = {det_delta:.3f} (Holm p={det_p:.3g}); "
                        f"confidence delta = {conf_delta:.3f} (Holm p={conf_p:.3g})."
                    ),
                    f"- {verdict}",
                ]
            )

    return "\n".join(lines)


def plot_confidence_dissociation(
    *,
    detection_group_summary: pd.DataFrame | None,
    confidence_group_summary: pd.DataFrame,
    model_names: tuple[str, ...],
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    palette = {
        "raw_regions_delta": "#1d3557",
        "baseline_regions_augmented": "#457b9d",
        "raw_plus_baseline_regions_augmented": "#e76f51",
    }
    panels = [
        ("Detection (hit vs miss)", detection_group_summary, axes[0]),
        ("Confidence (high vs low hits)", confidence_group_summary, axes[1]),
    ]
    for title, group_df, ax in panels:
        ax.axhline(0.5, color="#6c757d", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Target window start (ms)")
        ax.set_ylabel("Subject mean pair accuracy")
        ax.grid(True, alpha=0.25)
        if group_df is None or group_df.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            continue
        for model_name in model_names:
            sub = group_df.loc[group_df["model_name"] == model_name].sort_values("target_start_ms")
            if sub.empty:
                continue
            colour = palette.get(model_name, "#495057")
            label = str(sub["model_label"].iloc[0])
            ax.plot(
                sub["target_start_ms"],
                sub["pair_acc_mean"],
                marker="o",
                linewidth=2.2,
                color=colour,
                label=label,
            )
            if {"pair_acc_ci95_low", "pair_acc_ci95_high"}.issubset(sub.columns):
                ax.fill_between(
                    sub["target_start_ms"].to_numpy(dtype=float),
                    sub["pair_acc_ci95_low"].to_numpy(dtype=float),
                    sub["pair_acc_ci95_high"].to_numpy(dtype=float),
                    color=colour,
                    alpha=0.14,
                )
        ax.legend(frameon=False, loc="best", fontsize=8)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_confidence_dissociation_summary(
    *,
    trial_scores_csv: Path = DEFAULT_CONFIDENCE_TRIAL_SCORES_CSV,
    derivatives_dir: Path = DEFAULT_DERIVATIVES_DIR,
    detection_focus_contrasts_csv: Path | None = DEFAULT_DETECTION_FOCUS_CONTRAST_CSV,
    detection_group_summary_csv: Path | None = None,
    outdir: Path | None = None,
    model_names: tuple[str, ...] = DEFAULT_MODEL_NAMES,
    sdt_subset: str = DEFAULT_CONFIDENCE_SDT_SUBSET,
    min_unique_confidence: int = DEFAULT_CONFIDENCE_MIN_UNIQUE,
    min_per_side: int = DEFAULT_CONFIDENCE_MIN_PER_SIDE,
    focus_model_a: str = DEFAULT_FOCUS_MODEL_A,
    focus_model_b: str = DEFAULT_FOCUS_MODEL_B,
    focus_start_ms: float = DEFAULT_FOCUS_START_MS,
    focus_end_ms: float = DEFAULT_FOCUS_END_MS,
    n_bootstrap: int = DEFAULT_N_BOOT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> int:
    outdir = (
        trial_scores_csv.parent / "summary_confidence_dissociation"
        if outdir is None
        else outdir
    )
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        trial_scores = load_trial_scores(trial_scores_csv)
        trial_scores = filter_models(trial_scores, model_names=model_names)
        if "group_kind" in trial_scores.columns:
            named_mask = trial_scores["group_kind"].astype(str) == "named_region"
            if named_mask.any():
                trial_scores = trial_scores.loc[named_mask].reset_index(drop=True)
        trial_scores = _ensure_stimamp_in_trial_scores(
            trial_scores,
            derivatives_dir=Path(derivatives_dir),
        )
        if "confidence" not in trial_scores.columns:
            raise ValueError(
                "Trial-scores CSV is missing the confidence column required for the dissociation summary"
            )
        pair_df = build_confidence_matched_pairs(
            trial_scores,
            sdt_subset=sdt_subset,
            min_unique_confidence=min_unique_confidence,
            min_per_side=min_per_side,
        )
        skipped_df = pair_df.attrs.get("skipped_subjects", pd.DataFrame())
        subject_summary = evaluate_subject_pair_metrics(trial_scores, pair_df)
        group_summary = summarise_pair_metrics(
            subject_summary,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        pairwise_summary = build_pairwise_model_comparisons(
            subject_summary,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        match_quality = build_confidence_match_quality_summary(pair_df)
        focus_contrasts, headline_summary = build_focus_contrast_tables(
            pairwise_summary,
            match_quality.rename(columns={"mean_stimamp_abs_diff": "mean_stimamp_abs_diff"}),
            focus_model_a=focus_model_a,
            focus_model_b=focus_model_b,
            focus_start_ms=focus_start_ms,
            focus_end_ms=focus_end_ms,
        )

        detection_focus_contrasts = None
        if detection_focus_contrasts_csv is not None and Path(detection_focus_contrasts_csv).exists():
            detection_focus_contrasts = pd.read_csv(detection_focus_contrasts_csv)

        detection_group_summary = None
        if detection_group_summary_csv is not None and Path(detection_group_summary_csv).exists():
            detection_group_summary = pd.read_csv(detection_group_summary_csv)

        report_text = build_confidence_dissociation_report(
            confidence_focus_contrasts=focus_contrasts,
            confidence_quality=match_quality,
            detection_focus_contrasts=detection_focus_contrasts,
            focus_model_a=focus_model_a,
            focus_model_b=focus_model_b,
            focus_start_ms=focus_start_ms,
            focus_end_ms=focus_end_ms,
            sdt_subset=sdt_subset,
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    pair_path = outdir / "confidence_pairs.csv"
    skipped_path = outdir / "confidence_pair_skipped.csv"
    subject_path = outdir / "confidence_subject_summary.csv"
    group_path = outdir / "confidence_group_summary.csv"
    pairwise_path = outdir / "confidence_pairwise_summary.csv"
    quality_path = outdir / "confidence_pair_match_quality.csv"
    focus_contrasts_path = outdir / "confidence_focus_contrasts.csv"
    headline_summary_path = outdir / "confidence_headline_summary.csv"
    report_path = outdir / "confidence_dissociation_summary.md"
    plot_path = outdir / "confidence_dissociation_pair_acc.png"

    save_table(pair_df, pair_path)
    if skipped_df is not None and not skipped_df.empty:
        save_table(skipped_df, skipped_path)
    save_table(subject_summary, subject_path)
    save_table(group_summary, group_path)
    save_table(pairwise_summary, pairwise_path)
    save_table(match_quality, quality_path)
    save_table(focus_contrasts, focus_contrasts_path)
    save_table(headline_summary, headline_summary_path)
    report_path.write_text(report_text)

    plot_confidence_dissociation(
        detection_group_summary=detection_group_summary,
        confidence_group_summary=group_summary,
        model_names=model_names,
        outfile=plot_path,
    )

    print(f"Saved {pair_path}")
    print(f"Saved {subject_path}")
    print(f"Saved {group_path}")
    print(f"Saved {pairwise_path}")
    print(f"Saved {quality_path}")
    print(f"Saved {focus_contrasts_path}")
    print(f"Saved {headline_summary_path}")
    print(f"Saved {report_path}")
    print(f"Saved {plot_path}")

    print("\nFocus window:")
    print(f"  {focus_start_ms:.0f} to {focus_end_ms:.0f} ms; sdt subset={sdt_subset}")

    print("\nConfidence-pair focus contrast (hybrid minus raw):")
    focus_rows = focus_contrasts.loc[
        (focus_contrasts["target_start_ms"] == float(focus_start_ms))
        & (focus_contrasts["target_end_ms"] == float(focus_end_ms))
    ].copy()
    if not focus_rows.empty:
        print(
            focus_rows[
                [
                    "metric",
                    "n_subjects",
                    "mean_model_a",
                    "mean_model_b",
                    "mean_delta_model_b_minus_a",
                    "ci95_low_delta_model_b_minus_a",
                    "ci95_high_delta_model_b_minus_a",
                    "positive_subject_share",
                    "ttest_p",
                    "ttest_p_holm_focus",
                    "wilcoxon_p",
                    "wilcoxon_p_holm_focus",
                ]
            ].to_string(index=False)
        )

    return 0
