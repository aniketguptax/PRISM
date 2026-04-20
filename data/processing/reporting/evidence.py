"""Held-out evidence analysis for hit-versus-miss decoding and confidence."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from eegprep import load_exported_subject, load_subject_channel_labels
from experiments.spatial import build_channel_groups
from experiments.temporal import infer_boundary_tolerance_ms, slice_trial_window
from framework.summaries import prepare_pyplot, save_table
from reporting.behaviour import DEFAULT_DERIVATIVES_DIR, load_all_epoch_metadata


DEFAULT_REGIONS = ("central", "frontal")
DEFAULT_POSITIVE_SDT = "hit"
DEFAULT_NEGATIVE_SDT = "miss"
DEFAULT_BASELINE_PATTERN = "sub-*_region_sliding_baseline.csv"
DEFAULT_PRISM_PATTERN = "sub-*_region_window_prism.csv"
DEFAULT_EXPORT_DIR = Path("./data/exports_mat")
FEATURE_PREFIXES = ("baseline_", "prism_", "raw_")


@dataclass(frozen=True)
class EvidenceModelSpec:
    name: str
    label: str
    source: str
    regions: tuple[str, ...]
    metrics: tuple[str, ...]
    include_raw: bool = False


DEFAULT_MODEL_SPECS = (
    EvidenceModelSpec(
        name="raw_central_delta",
        label="Raw central baseline-corrected EEG",
        source="raw",
        regions=("central",),
        metrics=(),
    ),
    EvidenceModelSpec(
        name="raw_frontal_delta",
        label="Raw frontal baseline-corrected EEG",
        source="raw",
        regions=("frontal",),
        metrics=(),
    ),
    EvidenceModelSpec(
        name="raw_cf_delta",
        label="Raw central+frontal baseline-corrected EEG",
        source="raw",
        regions=("central", "frontal"),
        metrics=(),
    ),
    EvidenceModelSpec(
        name="baseline_central_r2",
        label="Baseline central R^2",
        source="baseline",
        regions=("central",),
        metrics=("pred_r2_obs",),
    ),
    EvidenceModelSpec(
        name="baseline_frontal_r2",
        label="Baseline frontal R^2",
        source="baseline",
        regions=("frontal",),
        metrics=("pred_r2_obs",),
    ),
    EvidenceModelSpec(
        name="baseline_cf_r2",
        label="Baseline central+frontal R^2",
        source="baseline",
        regions=("central", "frontal"),
        metrics=("pred_r2_obs",),
    ),
    EvidenceModelSpec(
        name="baseline_cf_augmented",
        label="Baseline central+frontal augmented",
        source="baseline",
        regions=("central", "frontal"),
        metrics=("pred_r2_obs", "pred_mse_obs", "pred_r2_latent", "pred_nll_latent"),
    ),
    EvidenceModelSpec(
        name="prism_central_r2",
        label="PRISM central R^2",
        source="prism",
        regions=("central",),
        metrics=("pred_r2_obs",),
    ),
    EvidenceModelSpec(
        name="prism_frontal_r2",
        label="PRISM frontal R^2",
        source="prism",
        regions=("frontal",),
        metrics=("pred_r2_obs",),
    ),
    EvidenceModelSpec(
        name="prism_cf_r2",
        label="PRISM central+frontal R^2",
        source="prism",
        regions=("central", "frontal"),
        metrics=("pred_r2_obs",),
    ),
    EvidenceModelSpec(
        name="prism_cf_augmented",
        label="PRISM central+frontal augmented",
        source="prism",
        regions=("central", "frontal"),
        metrics=("pred_r2_obs", "macro_logloss", "n_macro_states"),
    ),
    EvidenceModelSpec(
        name="raw_cf_plus_baseline_cf_r2",
        label="Raw central+frontal EEG + baseline R^2",
        source="baseline",
        regions=("central", "frontal"),
        metrics=("pred_r2_obs",),
        include_raw=True,
    ),
    EvidenceModelSpec(
        name="raw_cf_plus_baseline_cf_augmented",
        label="Raw central+frontal EEG + baseline augmented",
        source="baseline",
        regions=("central", "frontal"),
        metrics=("pred_r2_obs", "pred_mse_obs", "pred_r2_latent", "pred_nll_latent"),
        include_raw=True,
    ),
    EvidenceModelSpec(
        name="raw_cf_plus_prism_cf_r2",
        label="Raw central+frontal EEG + PRISM R^2",
        source="prism",
        regions=("central", "frontal"),
        metrics=("pred_r2_obs",),
        include_raw=True,
    ),
    EvidenceModelSpec(
        name="raw_cf_plus_prism_cf_augmented",
        label="Raw central+frontal EEG + PRISM augmented",
        source="prism",
        regions=("central", "frontal"),
        metrics=("pred_r2_obs", "macro_logloss", "n_macro_states"),
        include_raw=True,
    ),
)

DEFAULT_MODEL_ORDER = [spec.name for spec in DEFAULT_MODEL_SPECS]


def _load_result_set(
    results_dir: Path,
    *,
    pattern: str,
    combined_filename: str,
) -> pd.DataFrame:
    subject_files = sorted(results_dir.glob(pattern))
    if subject_files:
        frames = [pd.read_csv(path) for path in subject_files]
        return pd.concat(frames, ignore_index=True)

    combined_path = results_dir / combined_filename
    if combined_path.exists():
        return pd.read_csv(combined_path)

    raise FileNotFoundError(f"No matching result files found in {results_dir}")


def load_baseline_region_results(results_dir: Path) -> pd.DataFrame:
    df = _load_result_set(
        results_dir,
        pattern=DEFAULT_BASELINE_PATTERN,
        combined_filename="all_subjects_region_sliding_baseline.csv",
    )
    required = {"subject", "trial_idx", "region_name", "rep_dim", "pred_r2_obs"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Baseline region results are missing required columns: {missing}")
    if "error" in df.columns:
        df = df.loc[df["error"].fillna("").astype(str).eq("")].copy()
    return df.reset_index(drop=True)


def load_prism_region_results(results_dir: Path) -> pd.DataFrame:
    df = _load_result_set(
        results_dir,
        pattern=DEFAULT_PRISM_PATTERN,
        combined_filename="all_subjects_region_window_prism.csv",
    )
    required = {
        "subject",
        "trial_idx",
        "region_name",
        "model_family",
        "rep_dim",
        "pred_r2_obs",
        "macro_logloss",
        "n_macro_states",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"PRISM region results are missing required columns: {missing}")
    if "error" in df.columns:
        df = df.loc[df["error"].fillna("").astype(str).eq("")].copy()
    return df.reset_index(drop=True)


def filter_baseline_rows(
    df: pd.DataFrame,
    *,
    rep_dim: int,
    regions: tuple[str, ...],
    test_start_ms: float,
    test_end_ms: float,
) -> pd.DataFrame:
    filtered = df.loc[
        (df["rep_dim"] == int(rep_dim))
        & (df["region_name"].isin(regions))
        & (df["target_start_ms"] == float(test_start_ms))
        & (df["target_end_ms"] == float(test_end_ms))
    ].copy()
    if filtered.empty:
        raise ValueError("No baseline rows matched the requested evidence window")
    return filtered


def filter_prism_rows(
    df: pd.DataFrame,
    *,
    rep_dim: int,
    regions: tuple[str, ...],
    test_start_ms: float,
    test_end_ms: float,
    prism_model_family: str,
) -> pd.DataFrame:
    filtered = df.loc[
        (df["rep_dim"] == int(rep_dim))
        & (df["region_name"].isin(regions))
        & (df["test_start_ms"] == float(test_start_ms))
        & (df["test_end_ms"] == float(test_end_ms))
        & (df["model_family"] == prism_model_family)
    ].copy()
    if filtered.empty:
        raise ValueError("No PRISM rows matched the requested evidence window")
    return filtered


def build_feature_frame(
    df: pd.DataFrame,
    *,
    regions: tuple[str, ...],
    metrics: tuple[str, ...],
    prefix: str,
) -> pd.DataFrame:
    base = df.loc[df["region_name"].isin(regions)].copy()
    frames = []
    for metric in metrics:
        if metric not in base.columns:
            raise ValueError(f"Requested metric {metric!r} is not available")
        wide = base.pivot_table(
            index=["subject", "trial_idx"],
            columns="region_name",
            values=metric,
            aggfunc="first",
        ).reindex(columns=list(regions))
        wide.columns = [f"{prefix}_{region}_{metric}" for region in wide.columns]
        frames.append(wide)

    feature_df = pd.concat(frames, axis=1).reset_index()
    return feature_df


def collect_common_trials(
    baseline_df: pd.DataFrame,
    prism_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return the subject-trial pairs available in both result sets."""
    baseline_trials = baseline_df[["subject", "trial_idx"]].drop_duplicates()
    prism_trials = prism_df[["subject", "trial_idx"]].drop_duplicates()
    common_trials = baseline_trials.merge(
        prism_trials,
        on=["subject", "trial_idx"],
        how="inner",
        validate="one_to_one",
    )
    if common_trials.empty:
        raise ValueError("No overlapping trials were found between baseline and PRISM")
    return common_trials.sort_values(["subject", "trial_idx"]).reset_index(drop=True)


def require_single_window_value(
    df: pd.DataFrame,
    column: str,
    *,
    label: str,
) -> float:
    """Extract one shared window bound from a filtered result table."""
    values = sorted(df[column].dropna().astype(float).unique().tolist())
    if len(values) != 1:
        raise ValueError(f"Expected one {label}, found {values}")
    return float(values[0])


def build_raw_feature_frame(
    common_trials: pd.DataFrame,
    *,
    export_dir: Path,
    derivatives_dir: Path,
    baseline_start_ms: float,
    baseline_end_ms: float,
    test_start_ms: float,
    test_end_ms: float,
    regions: tuple[str, ...],
) -> pd.DataFrame:
    """Build baseline-corrected regional EEG features from the exported trials."""
    rows: list[dict] = []

    for subject, subject_trials in common_trials.groupby("subject", observed=False):
        subject_path = export_dir / f"{subject}_preproc_01hz_export.mat"
        data, _sfreq, times_ms = load_exported_subject(subject_path)
        channel_labels = load_subject_channel_labels(subject, derivatives_dir=derivatives_dir)
        if data.shape[2] != len(channel_labels):
            raise ValueError(
                f"Channel count mismatch for {subject}: data has {data.shape[2]} channels "
                f"but metadata has {len(channel_labels)} labels"
            )

        region_groups = build_channel_groups(channel_labels)
        missing_regions = [region for region in regions if region not in region_groups]
        if missing_regions:
            raise ValueError(f"{subject} is missing requested regions: {missing_regions}")

        boundary_tolerance_ms = infer_boundary_tolerance_ms(times_ms)
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
                test_start_ms,
                test_end_ms,
                require_full_window=True,
                boundary_tolerance_ms=boundary_tolerance_ms,
            )
            delta = target_X.mean(axis=0) - baseline_X.mean(axis=0)

            row = {"subject": subject, "trial_idx": int(trial_idx)}
            for region in regions:
                region_idx = np.asarray(region_groups[region], dtype=int)
                for chan_idx in region_idx:
                    label = str(channel_labels[int(chan_idx)]).strip()
                    row[f"raw_{region}__{label}"] = float(delta[int(chan_idx)])
            rows.append(row)

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError("No raw EEG feature rows could be constructed")

    feature_cols = [
        column
        for column in raw_df.columns
        if column.startswith("raw_") and raw_df[column].notna().all()
    ]
    if not feature_cols:
        raise ValueError("No common raw EEG feature columns were available across subjects")

    return raw_df.loc[:, ["subject", "trial_idx", *sorted(feature_cols)]]


def select_raw_feature_frame(
    raw_feature_df: pd.DataFrame,
    *,
    regions: tuple[str, ...],
) -> pd.DataFrame:
    """Select the raw EEG columns for the requested region set."""
    keep = ["subject", "trial_idx"]
    for region in regions:
        region_cols = [
            column
            for column in raw_feature_df.columns
            if column.startswith(f"raw_{region}__")
        ]
        if not region_cols:
            raise ValueError(f"No raw EEG columns were available for region {region!r}")
        keep.extend(region_cols)
    return raw_feature_df.loc[:, keep].copy()


def merge_feature_frames(feature_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge one or more feature tables on subject and trial index."""
    if not feature_frames:
        raise ValueError("At least one feature frame is required")

    merged = feature_frames[0].copy()
    for frame in feature_frames[1:]:
        merged = merged.merge(
            frame,
            on=["subject", "trial_idx"],
            how="inner",
            validate="one_to_one",
        )
    return merged


def build_model_feature_frame(
    spec: EvidenceModelSpec,
    *,
    baseline_df: pd.DataFrame,
    prism_df: pd.DataFrame,
    raw_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Construct one evidence feature table for a single model spec."""
    feature_frames: list[pd.DataFrame] = []

    if spec.source == "raw" or spec.include_raw:
        feature_frames.append(
            select_raw_feature_frame(
                raw_feature_df,
                regions=spec.regions,
            )
        )

    if spec.source == "baseline":
        feature_frames.append(
            build_feature_frame(
                baseline_df,
                regions=spec.regions,
                metrics=spec.metrics,
                prefix="baseline",
            )
        )
    elif spec.source == "prism":
        feature_frames.append(
            build_feature_frame(
                prism_df,
                regions=spec.regions,
                metrics=spec.metrics,
                prefix="prism",
            )
        )
    elif spec.source != "raw":
        raise ValueError(f"Unknown evidence model source: {spec.source!r}")

    return merge_feature_frames(feature_frames)


def load_signal_metadata(
    subjects: list[str],
    *,
    derivatives_dir: Path,
    positive_sdt: str,
    negative_sdt: str,
) -> pd.DataFrame:
    metadata = load_all_epoch_metadata(subjects, derivatives_dir=derivatives_dir)
    metadata = metadata.loc[metadata["sdt"].isin([positive_sdt, negative_sdt])].copy()
    if metadata.empty:
        raise ValueError("No signal-present trials were found in the metadata")
    metadata["label"] = (metadata["sdt"] == positive_sdt).astype(int)
    return metadata


def assign_subject_twofold_splits(
    trial_df: pd.DataFrame,
    *,
    label_col: str = "label",
) -> pd.DataFrame:
    trial_df = trial_df.sort_values(["subject", label_col, "trial_idx"]).copy()
    trial_df["fold"] = -1
    for (_, _), group_df in trial_df.groupby(["subject", label_col], observed=False):
        idx = group_df.sort_values("trial_idx").index.to_numpy()
        trial_df.loc[idx, "fold"] = np.arange(idx.size) % 2
    return trial_df


def build_model_trial_table(
    metadata_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    *,
    model_name: str,
    model_label: str,
) -> pd.DataFrame:
    merged = metadata_df.merge(
        feature_df,
        on=["subject", "trial_idx"],
        how="inner",
        validate="one_to_one",
    )
    feature_cols = [col for col in merged.columns if col.startswith(FEATURE_PREFIXES)]
    merged = merged.dropna(subset=feature_cols).copy()
    merged["model_name"] = model_name
    merged["model_label"] = model_label
    return merged


def subject_centre_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    subject_means = train_df.groupby("subject", observed=False)[feature_cols].mean()
    global_mean = train_df[feature_cols].mean()

    def centre(frame: pd.DataFrame) -> np.ndarray:
        means = subject_means.reindex(frame["subject"]).fillna(global_mean)
        return frame[feature_cols].to_numpy(dtype=float) - means.to_numpy(dtype=float)

    return centre(train_df), centre(test_df)


def standardise_features(
    train_X: np.ndarray,
    test_X: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_X.mean(axis=0, keepdims=True)
    scale = train_X.std(axis=0, keepdims=True)
    keep = (scale.reshape(-1) > eps).astype(bool)
    if not np.any(keep):
        raise ValueError("All evidence features were constant in the training fold")
    mean = mean[:, keep]
    scale = scale[:, keep]
    return (train_X[:, keep] - mean) / scale, (test_X[:, keep] - mean) / scale, keep


def fit_fisher_discriminant(
    train_X: np.ndarray,
    train_y: np.ndarray,
    *,
    ridge: float = 1e-4,
) -> tuple[np.ndarray, float]:
    train_y = np.asarray(train_y, dtype=int)
    pos = train_X[train_y == 1]
    neg = train_X[train_y == 0]
    if pos.shape[0] < 2 or neg.shape[0] < 2:
        raise ValueError("Both classes need at least two training samples per fold")

    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)
    pos_cov = np.atleast_2d(np.cov(pos, rowvar=False, bias=False))
    neg_cov = np.atleast_2d(np.cov(neg, rowvar=False, bias=False))
    pooled = ((pos.shape[0] - 1) * pos_cov + (neg.shape[0] - 1) * neg_cov) / (
        pos.shape[0] + neg.shape[0] - 2
    )
    pooled = np.asarray(pooled, dtype=float)
    pooled = 0.5 * (pooled + pooled.T)
    pooled.flat[:: pooled.shape[0] + 1] += ridge

    try:
        weights = np.linalg.solve(pooled, pos_mean - neg_mean)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(pooled) @ (pos_mean - neg_mean)

    log_prior = np.log(pos.shape[0] / neg.shape[0])
    intercept = float(-0.5 * (pos_mean + neg_mean) @ weights + log_prior)
    return weights, intercept


def roc_auc_score_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    pos_ranks = float(np.sum(ranks[y_true == 1]))
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_subject_metrics(
    trial_scores: pd.DataFrame,
    *,
    positive_sdt: str,
) -> pd.DataFrame:
    rows = []
    for (model_name, model_label, subject), subject_df in trial_scores.groupby(
        ["model_name", "model_label", "subject"],
        observed=False,
    ):
        y_true = subject_df["label"].to_numpy(dtype=int)
        scores = subject_df["evidence_score"].to_numpy(dtype=float)
        preds = (scores >= 0.0).astype(int)
        hit_df = subject_df.loc[
            (subject_df["sdt"] == positive_sdt) & subject_df["confidence"].notna()
        ].copy()
        if hit_df.shape[0] >= 3 and hit_df["confidence"].nunique() >= 2:
            hit_rho, hit_p = stats.spearmanr(hit_df["confidence"], hit_df["evidence_score"])
            hit_rho = float(hit_rho)
            hit_p = float(hit_p)
        else:
            hit_rho = float("nan")
            hit_p = float("nan")

        rows.append(
            {
                "model_name": model_name,
                "model_label": model_label,
                "subject": subject,
                "n_trials": int(subject_df.shape[0]),
                "n_hits": int(np.sum(y_true == 1)),
                "n_misses": int(np.sum(y_true == 0)),
                "auc": roc_auc_score_binary(y_true, scores),
                "accuracy": float(np.mean(preds == y_true)),
                "mean_hit_score": float(subject_df.loc[subject_df["label"] == 1, "evidence_score"].mean()),
                "mean_miss_score": float(subject_df.loc[subject_df["label"] == 0, "evidence_score"].mean()),
                "hit_conf_rho": hit_rho,
                "hit_conf_p": hit_p,
            }
        )

    return pd.DataFrame(rows)


def summarise_model_metrics(
    trial_scores: pd.DataFrame,
    subject_summary: pd.DataFrame,
    *,
    positive_sdt: str,
) -> pd.DataFrame:
    rows = []
    for (model_name, model_label), group_df in subject_summary.groupby(
        ["model_name", "model_label"],
        observed=False,
    ):
        auc_values = group_df["auc"].dropna().to_numpy(dtype=float)
        rho_values = group_df["hit_conf_rho"].dropna().to_numpy(dtype=float)
        pooled_df = trial_scores.loc[trial_scores["model_name"] == model_name].copy()
        pooled_auc = roc_auc_score_binary(
            pooled_df["label"].to_numpy(dtype=int),
            pooled_df["evidence_score"].to_numpy(dtype=float),
        )
        pooled_hits = pooled_df.loc[
            (pooled_df["sdt"] == positive_sdt) & pooled_df["confidence"].notna()
        ].copy()
        if pooled_hits.shape[0] >= 3 and pooled_hits["confidence"].nunique() >= 2:
            pooled_hit_rho, pooled_hit_p = stats.spearmanr(
                pooled_hits["confidence"],
                pooled_hits["evidence_score"],
            )
            pooled_hit_rho = float(pooled_hit_rho)
            pooled_hit_p = float(pooled_hit_p)
        else:
            pooled_hit_rho = float("nan")
            pooled_hit_p = float("nan")

        auc_t = (
            stats.ttest_1samp(auc_values, 0.5, nan_policy="omit")
            if auc_values.size >= 2
            else None
        )
        rho_t = (
            stats.ttest_1samp(rho_values, 0.0, nan_policy="omit")
            if rho_values.size >= 2
            else None
        )
        rows.append(
            {
                "model_name": model_name,
                "model_label": model_label,
                "n_subjects_auc": int(auc_values.size),
                "auc_mean": float(np.mean(auc_values)) if auc_values.size else float("nan"),
                "auc_std": float(np.std(auc_values, ddof=1)) if auc_values.size > 1 else float("nan"),
                "auc_above_half_share": float(np.mean(auc_values > 0.5)) if auc_values.size else float("nan"),
                "auc_ttest_p": float(auc_t.pvalue) if auc_t is not None else float("nan"),
                "n_subjects_hit_conf": int(rho_values.size),
                "hit_conf_rho_mean": float(np.mean(rho_values)) if rho_values.size else float("nan"),
                "hit_conf_rho_std": float(np.std(rho_values, ddof=1)) if rho_values.size > 1 else float("nan"),
                "hit_conf_positive_share": float(np.mean(rho_values > 0.0)) if rho_values.size else float("nan"),
                "hit_conf_ttest_p": float(rho_t.pvalue) if rho_t is not None else float("nan"),
                "pooled_auc": pooled_auc,
                "pooled_hit_conf_rho": pooled_hit_rho,
                "pooled_hit_conf_p": pooled_hit_p,
                "pooled_trials": int(pooled_df.shape[0]),
                "pooled_hits": int(np.sum(pooled_df["label"] == 1)),
                "pooled_misses": int(np.sum(pooled_df["label"] == 0)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    present_names = out["model_name"].dropna().astype(str).tolist()
    ordered_names = [name for name in DEFAULT_MODEL_ORDER if name in present_names]
    ordered_names.extend(name for name in present_names if name not in ordered_names)
    out["model_name"] = pd.Categorical(
        out["model_name"],
        categories=ordered_names,
        ordered=True,
    )
    return out.sort_values("model_name").reset_index(drop=True)


def build_pairwise_model_comparisons(subject_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = sorted(subject_summary["model_name"].dropna().astype(str).unique().tolist())
    for model_a, model_b in combinations(models, 2):
        wide = (
            subject_summary.loc[subject_summary["model_name"].isin([model_a, model_b])]
            .pivot(index="subject", columns="model_name", values=["auc", "hit_conf_rho"])
        )
        for metric, null_value in (("auc", 0.0), ("hit_conf_rho", 0.0)):
            if metric not in wide.columns.get_level_values(0):
                continue
            metric_wide = wide[metric]
            if model_a not in metric_wide.columns or model_b not in metric_wide.columns:
                continue
            pair_df = metric_wide[[model_a, model_b]].dropna()
            if pair_df.empty:
                continue
            delta = pair_df[model_b] - pair_df[model_a]
            if pair_df.shape[0] >= 2:
                t_stat, t_p = stats.ttest_1samp(delta.to_numpy(dtype=float), null_value)
                t_stat = float(t_stat)
                t_p = float(t_p)
            else:
                t_stat = float("nan")
                t_p = float("nan")
            rows.append(
                {
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_subjects": int(pair_df.shape[0]),
                    "mean_model_a": float(pair_df[model_a].mean()),
                    "mean_model_b": float(pair_df[model_b].mean()),
                    "mean_delta_model_b_minus_a": float(delta.mean()),
                    "positive_subject_share": float(np.mean(delta > 0.0)),
                    "ttest_stat": t_stat,
                    "ttest_p": t_p,
                }
            )
    return pd.DataFrame(rows)


def fit_out_of_fold_evidence(
    trial_df: pd.DataFrame,
    *,
    feature_cols: list[str],
) -> pd.DataFrame:
    predictions = []
    for fold_idx in (0, 1):
        train_df = trial_df.loc[trial_df["fold"] != fold_idx].copy()
        test_df = trial_df.loc[trial_df["fold"] == fold_idx].copy()
        if train_df.empty or test_df.empty:
            continue
        if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
            continue

        train_X, test_X = subject_centre_features(
            train_df,
            test_df,
            feature_cols=feature_cols,
        )
        train_X, test_X, keep = standardise_features(train_X, test_X)
        _ = keep
        weights, intercept = fit_fisher_discriminant(
            train_X,
            train_df["label"].to_numpy(dtype=int),
        )
        train_scores = train_X @ weights + intercept
        if np.mean(train_scores[train_df["label"].to_numpy(dtype=int) == 1]) < np.mean(
            train_scores[train_df["label"].to_numpy(dtype=int) == 0]
        ):
            weights = -weights
            intercept = -intercept

        test_scores = test_X @ weights + intercept
        fold_out = test_df[
            [
                "subject",
                "trial_idx",
                "sdt",
                "confidence",
                "label",
                "fold",
                "model_name",
                "model_label",
            ]
        ].copy()
        fold_out["evidence_score"] = test_scores
        predictions.append(fold_out)

    if not predictions:
        raise ValueError("No valid out-of-fold predictions could be formed")
    return pd.concat(predictions, ignore_index=True).sort_values(
        ["model_name", "subject", "trial_idx"]
    ).reset_index(drop=True)


def plot_evidence_summary(group_summary: pd.DataFrame, outfile: Path) -> None:
    if group_summary.empty:
        return

    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(group_summary.shape[0], dtype=float)

    axes[0].bar(x, group_summary["auc_mean"], color="#0f4c5c", alpha=0.9)
    axes[0].errorbar(
        x,
        group_summary["auc_mean"],
        yerr=group_summary["auc_std"],
        fmt="none",
        ecolor="#1f2d3d",
        capsize=3,
    )
    axes[0].axhline(0.5, color="#495057", linestyle="--", linewidth=1.0)
    axes[0].set_title("Held-out hit versus miss AUC")
    axes[0].set_ylabel("Subject mean AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(group_summary["model_name"], rotation=35, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, group_summary["hit_conf_rho_mean"], color="#e36414", alpha=0.9)
    axes[1].errorbar(
        x,
        group_summary["hit_conf_rho_mean"],
        yerr=group_summary["hit_conf_rho_std"],
        fmt="none",
        ecolor="#6f1d1b",
        capsize=3,
    )
    axes[1].axhline(0.0, color="#495057", linestyle="--", linewidth=1.0)
    axes[1].set_title("Held-out hit confidence association")
    axes[1].set_ylabel("Subject mean Spearman rho")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(group_summary["model_name"], rotation=35, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_evidence_report(
    group_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Evidence Summary",
        "",
    ]
    if not group_summary.empty:
        best_auc = group_summary.sort_values("pooled_auc", ascending=False).iloc[0]
        lines.extend(
            [
                "## Detection",
                "",
                (
                    f"- Strongest pooled held-out AUC: {best_auc['model_name']} "
                    f"(pooled AUC={best_auc['pooled_auc']:.3f}, "
                    f"subject mean AUC={best_auc['auc_mean']:.3f})."
                ),
                "",
            ]
        )

        best_conf = group_summary.sort_values("pooled_hit_conf_rho", ascending=False).iloc[0]
        lines.extend(
            [
                "## Confidence",
                "",
                (
                    f"- Strongest pooled hit-confidence association: {best_conf['model_name']} "
                    f"(pooled rho={best_conf['pooled_hit_conf_rho']:.3f}, "
                    f"subject mean rho={best_conf['hit_conf_rho_mean']:.3f})."
                ),
                "",
            ]
        )

    if not pairwise_summary.empty:
        lines.extend(["## Pairwise contrasts", ""])
        for metric, metric_df in pairwise_summary.groupby("metric", observed=False):
            best = metric_df.sort_values("mean_delta_model_b_minus_a", ascending=False).iloc[0]
            lines.append(
                f"- {metric}: largest positive model-b minus model-a delta was "
                f"{best['model_b']} over {best['model_a']} "
                f"(mean delta={best['mean_delta_model_b_minus_a']:.3f}, p={best['ttest_p']:.3g})."
            )
        lines.append("")

    if len(lines) == 2:
        lines.append("No valid evidence results were available.")
    return "\n".join(lines)


def run_evidence_summary(
    *,
    baseline_results_dir: Path,
    prism_results_dir: Path,
    derivatives_dir: Path = DEFAULT_DERIVATIVES_DIR,
    export_dir: Path = DEFAULT_EXPORT_DIR,
    outdir: Path | None = None,
    rep_dim: int = 4,
    regions: tuple[str, ...] = DEFAULT_REGIONS,
    test_start_ms: float = 125.0,
    test_end_ms: float = 375.0,
    positive_sdt: str = DEFAULT_POSITIVE_SDT,
    negative_sdt: str = DEFAULT_NEGATIVE_SDT,
    prism_model_family: str = "prism_pca",
) -> int:
    outdir = prism_results_dir / "summary" if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        baseline_df = load_baseline_region_results(baseline_results_dir)
        prism_df = load_prism_region_results(prism_results_dir)
        baseline_df = filter_baseline_rows(
            baseline_df,
            rep_dim=rep_dim,
            regions=regions,
            test_start_ms=test_start_ms,
            test_end_ms=test_end_ms,
        )
        prism_df = filter_prism_rows(
            prism_df,
            rep_dim=rep_dim,
            regions=regions,
            test_start_ms=test_start_ms,
            test_end_ms=test_end_ms,
            prism_model_family=prism_model_family,
        )
        common_trials = collect_common_trials(baseline_df, prism_df)
        baseline_df = baseline_df.merge(
            common_trials,
            on=["subject", "trial_idx"],
            how="inner",
            validate="many_to_one",
        )
        prism_df = prism_df.merge(
            common_trials,
            on=["subject", "trial_idx"],
            how="inner",
            validate="many_to_one",
        )

        baseline_start_ms = require_single_window_value(
            baseline_df,
            "train_start_ms",
            label="baseline train window start",
        )
        baseline_end_ms = require_single_window_value(
            baseline_df,
            "train_end_ms",
            label="baseline train window end",
        )
        prism_start_ms = require_single_window_value(
            prism_df,
            "train_start_ms",
            label="PRISM train window start",
        )
        prism_end_ms = require_single_window_value(
            prism_df,
            "train_end_ms",
            label="PRISM train window end",
        )
        if not np.isclose(baseline_start_ms, prism_start_ms) or not np.isclose(
            baseline_end_ms,
            prism_end_ms,
        ):
            raise ValueError(
                "Baseline and PRISM runs do not share the same training window, "
                "so the evidence comparison would be mismatched"
            )

        subjects = sorted(common_trials["subject"].dropna().astype(str).unique().tolist())
        if not subjects:
            raise ValueError("No overlapping subjects were found between baseline and PRISM")

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
            raise ValueError("No hit or miss trials remained after aligning the evidence inputs")
        metadata_df = assign_subject_twofold_splits(metadata_df)
        raw_feature_df = build_raw_feature_frame(
            common_trials,
            export_dir=export_dir,
            derivatives_dir=derivatives_dir,
            baseline_start_ms=baseline_start_ms,
            baseline_end_ms=baseline_end_ms,
            test_start_ms=test_start_ms,
            test_end_ms=test_end_ms,
            regions=regions,
        )

        model_tables = []
        for spec in DEFAULT_MODEL_SPECS:
            if any(region not in regions for region in spec.regions):
                continue
            model_frame = build_model_feature_frame(
                spec,
                baseline_df=baseline_df,
                prism_df=prism_df,
                raw_feature_df=raw_feature_df,
            )
            model_trial_df = build_model_trial_table(
                metadata_df,
                model_frame,
                model_name=spec.name,
                model_label=spec.label,
            )
            feature_cols = [
                col
                for col in model_trial_df.columns
                if col.startswith(FEATURE_PREFIXES)
            ]
            trial_scores = fit_out_of_fold_evidence(
                model_trial_df,
                feature_cols=feature_cols,
            )
            model_tables.append(trial_scores)

        if not model_tables:
            raise ValueError("No evidence model could be constructed from the requested inputs")

        trial_scores = pd.concat(model_tables, ignore_index=True)
        subject_summary = evaluate_subject_metrics(
            trial_scores,
            positive_sdt=positive_sdt,
        )
        group_summary = summarise_model_metrics(
            trial_scores,
            subject_summary,
            positive_sdt=positive_sdt,
        )
        pairwise_summary = build_pairwise_model_comparisons(subject_summary)
        report = build_evidence_report(group_summary, pairwise_summary)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    trial_scores_path = outdir / "evidence_trial_scores.csv"
    subject_summary_path = outdir / "evidence_subject_summary.csv"
    group_summary_path = outdir / "evidence_group_summary.csv"
    pairwise_summary_path = outdir / "evidence_pairwise_summary.csv"
    plot_path = outdir / "evidence_summary.png"
    report_path = outdir / "evidence_summary.md"

    save_table(trial_scores, trial_scores_path)
    save_table(subject_summary, subject_summary_path)
    save_table(group_summary, group_summary_path)
    save_table(pairwise_summary, pairwise_summary_path)
    plot_evidence_summary(group_summary, plot_path)
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"Saved {trial_scores_path}")
    print(f"Saved {subject_summary_path}")
    print(f"Saved {group_summary_path}")
    print(f"Saved {pairwise_summary_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {report_path}")

    print("\nGroup summary:")
    print(group_summary.to_string(index=False))
    if not pairwise_summary.empty:
        print("\nPairwise summary:")
        print(pairwise_summary.to_string(index=False))
    return 0
