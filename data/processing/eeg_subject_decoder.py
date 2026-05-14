from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _eeg_stats import _ci95, _format_p
from eegprep import load_exported_subject, load_subject_channel_labels
from experiments.spatial import build_channel_groups
from experiments.temporal import infer_boundary_tolerance_ms, slice_trial_window
from reporting.behaviour import load_all_epoch_metadata


DEFAULT_BASELINE_RESULTS = Path("data/results_baseline/region_sliding_baseline300ms_controls_focus_q4")
DEFAULT_PRISM_RESULTS = Path("data/results_prism/eeg_prism_central_full_pca_q4")
DEFAULT_DERIVATIVES_DIR = Path("data/ds001785/derivatives/eegprep")
DEFAULT_EXPORT_DIR = Path("data/exports_mat")
DEFAULT_OUTDIR = DEFAULT_PRISM_RESULTS / "summary_subject_decoder"

DEFAULT_REGION = "central"
DEFAULT_REP_DIM = 4
DEFAULT_TEST_START_MS = 125.0
DEFAULT_TEST_END_MS = 375.0
DEFAULT_BASELINE_START_MS = -300.0
DEFAULT_BASELINE_END_MS = 0.0
DEFAULT_PRISM_MODEL_FAMILY = "prism_pca"
DEFAULT_N_FOLDS = 5


FEATURE_PREFIXES = ("raw_", "var_", "prism_")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    prefixes: tuple[str, ...]


MODEL_SPECS = (
    ModelSpec("raw_central_delta", "Raw central EEG", ("raw_",)),
    ModelSpec("var_central_augmented", "VAR features", ("var_",)),
    ModelSpec("prism_central_augmented", "PRISM summaries", ("prism_",)),
    ModelSpec("raw_plus_var", "Raw EEG + VAR features", ("raw_", "var_")),
    ModelSpec("raw_plus_prism", "Raw EEG + PRISM summaries", ("raw_", "prism_")),
)


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    return out[np.isfinite(out)]


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    y_true = y_true[finite]
    scores = scores[finite]
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = stats.rankdata(scores)
    pos_ranks = float(np.sum(ranks[y_true == 1]))
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _load_result_set(results_dir: Path, *, pattern: str, combined_filename: str) -> pd.DataFrame:
    combined = results_dir / combined_filename
    if combined.exists():
        return pd.read_csv(combined)
    files = sorted(results_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern!r} or {combined_filename!r} found in {results_dir}")
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


def _filter_clean(df: pd.DataFrame) -> pd.DataFrame:
    if "error" not in df.columns:
        return df.copy()
    return df.loc[df["error"].fillna("").astype(str).eq("")].copy()


def load_baseline_features(
    results_dir: Path,
    *,
    region: str,
    rep_dim: int,
    test_start_ms: float,
    test_end_ms: float,
) -> pd.DataFrame:
    df = _load_result_set(
        results_dir,
        pattern="sub-*_region_sliding_baseline.csv",
        combined_filename="all_subjects_region_sliding_baseline.csv",
    )
    df = _filter_clean(df)
    mask = (
        df["region_name"].astype(str).eq(region)
        & df["rep_dim"].astype(int).eq(int(rep_dim))
        & df["target_start_ms"].astype(float).eq(float(test_start_ms))
        & df["target_end_ms"].astype(float).eq(float(test_end_ms))
    )
    if "group_kind" in df.columns:
        mask &= df["group_kind"].astype(str).eq("named_region")
    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise ValueError("No baseline rows matched the requested subject-decoder analysis.")

    metrics = ("pred_r2_obs", "pred_mse_obs", "pred_r2_latent", "pred_nll_latent")
    cols = ["subject", "trial_idx", *metrics]
    out = filtered.loc[:, cols].drop_duplicates(["subject", "trial_idx"]).copy()
    return out.rename(columns={metric: f"var_{metric}" for metric in metrics})


def load_prism_features(
    results_dir: Path,
    *,
    region: str,
    rep_dim: int,
    test_start_ms: float,
    test_end_ms: float,
    model_family: str,
) -> pd.DataFrame:
    df = _load_result_set(
        results_dir,
        pattern="sub-*_region_window_prism.csv",
        combined_filename="all_subjects_region_window_prism.csv",
    )
    df = _filter_clean(df)
    mask = (
        df["region_name"].astype(str).eq(region)
        & df["rep_dim"].astype(int).eq(int(rep_dim))
        & df["test_start_ms"].astype(float).eq(float(test_start_ms))
        & df["test_end_ms"].astype(float).eq(float(test_end_ms))
        & df["model_family"].astype(str).eq(model_family)
    )
    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise ValueError("No PRISM rows matched the requested subject-decoder analysis.")

    metrics = ("pred_r2_obs", "pred_mse_obs", "macro_logloss", "n_macro_states", "psi_opt")
    cols = ["subject", "trial_idx", *metrics]
    out = filtered.loc[:, cols].drop_duplicates(["subject", "trial_idx"]).copy()
    return out.rename(columns={metric: f"prism_{metric}" for metric in metrics})


def load_metadata(
    subjects: list[str],
    *,
    derivatives_dir: Path,
    positive_sdt: str,
    negative_sdt: str,
) -> pd.DataFrame:
    metadata = load_all_epoch_metadata(subjects, derivatives_dir=derivatives_dir)
    metadata = metadata.loc[metadata["sdt"].isin([positive_sdt, negative_sdt])].copy()
    metadata["label"] = metadata["sdt"].astype(str).eq(positive_sdt).astype(int)
    return metadata


def build_raw_features(
    trials: pd.DataFrame,
    *,
    export_dir: Path,
    derivatives_dir: Path,
    region: str,
    baseline_start_ms: float,
    baseline_end_ms: float,
    test_start_ms: float,
    test_end_ms: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        subject = str(subject)
        data, _sfreq, times_ms = load_exported_subject(export_dir / f"{subject}_preproc_01hz_export.mat")
        channel_labels = load_subject_channel_labels(subject, derivatives_dir=derivatives_dir)
        region_groups = build_channel_groups(channel_labels)
        if region not in region_groups:
            raise ValueError(f"{subject} is missing channel region {region!r}.")
        region_idx = np.asarray(region_groups[region], dtype=int)
        tolerance_ms = infer_boundary_tolerance_ms(times_ms)

        for trial_idx in subject_trials["trial_idx"].astype(int).tolist():
            trial = np.asarray(data[trial_idx], dtype=float)
            baseline, _ = slice_trial_window(
                trial,
                times_ms,
                baseline_start_ms,
                baseline_end_ms,
                require_full_window=True,
                boundary_tolerance_ms=tolerance_ms,
            )
            target, _ = slice_trial_window(
                trial,
                times_ms,
                test_start_ms,
                test_end_ms,
                require_full_window=True,
                boundary_tolerance_ms=tolerance_ms,
            )
            delta = target.mean(axis=0) - baseline.mean(axis=0)
            row: dict[str, object] = {"subject": subject, "trial_idx": int(trial_idx)}
            for chan_idx in region_idx:
                label = str(channel_labels[int(chan_idx)]).strip()
                row[f"raw_{region}_{label}"] = float(delta[int(chan_idx)])
            rows.append(row)

    if not rows:
        raise ValueError("No raw EEG features could be built.")
    return pd.DataFrame(rows)


def assign_subject_folds(df: pd.DataFrame, *, n_folds: int) -> pd.DataFrame:
    out = df.sort_values(["subject", "label", "trial_idx"]).copy()
    out["decoder_fold"] = -1
    for (_, label), group in out.groupby(["subject", "label"], sort=True):
        idx = group.sort_values("trial_idx").index.to_numpy()
        out.loc[idx, "decoder_fold"] = np.arange(idx.size) % int(n_folds)
    return out.sort_values(["subject", "trial_idx"]).reset_index(drop=True)


def standardise_features(train_x: np.ndarray, test_x: np.ndarray, *, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    keep = scale.reshape(-1) > eps
    if not np.any(keep):
        raise ValueError("All decoder features were constant in a training fold.")
    return (train_x[:, keep] - mean[:, keep]) / scale[:, keep], (test_x[:, keep] - mean[:, keep]) / scale[:, keep]


def fit_fisher(train_x: np.ndarray, train_y: np.ndarray, *, ridge: float) -> tuple[np.ndarray, float]:
    train_y = np.asarray(train_y, dtype=int)
    pos = train_x[train_y == 1]
    neg = train_x[train_y == 0]
    if pos.shape[0] < 2 or neg.shape[0] < 2:
        raise ValueError("Both classes need at least two training samples.")

    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)
    pos_cov = np.atleast_2d(np.cov(pos, rowvar=False, bias=False))
    neg_cov = np.atleast_2d(np.cov(neg, rowvar=False, bias=False))
    pooled = ((pos.shape[0] - 1) * pos_cov + (neg.shape[0] - 1) * neg_cov) / (
        pos.shape[0] + neg.shape[0] - 2
    )
    pooled = np.asarray(pooled, dtype=float)
    pooled = 0.5 * (pooled + pooled.T)
    pooled.flat[:: pooled.shape[0] + 1] += float(ridge)

    try:
        weights = np.linalg.solve(pooled, pos_mean - neg_mean)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(pooled) @ (pos_mean - neg_mean)
    intercept = float(-0.5 * (pos_mean + neg_mean) @ weights + np.log(pos.shape[0] / neg.shape[0]))
    return weights, intercept


def decode_subject(
    subject_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    n_folds: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full(subject_df.shape[0], np.nan, dtype=float)
    folds = subject_df["decoder_fold"].to_numpy(dtype=int)
    y = subject_df["label"].to_numpy(dtype=int)

    for fold in range(n_folds):
        train_mask = folds != fold
        test_mask = folds == fold
        if not np.any(test_mask) or np.unique(y[train_mask]).size < 2:
            continue
        train_x = subject_df.loc[train_mask, feature_cols].to_numpy(dtype=float)
        test_x = subject_df.loc[test_mask, feature_cols].to_numpy(dtype=float)
        train_x, test_x = standardise_features(train_x, test_x)
        weights, intercept = fit_fisher(train_x, y[train_mask], ridge=ridge)
        train_scores = train_x @ weights + intercept
        if train_scores[y[train_mask] == 1].mean() < train_scores[y[train_mask] == 0].mean():
            weights = -weights
            intercept = -intercept
        scores[test_mask] = test_x @ weights + intercept
    return scores, y


def decode_all_subjects(
    frame: pd.DataFrame,
    *,
    model_specs: tuple[ModelSpec, ...],
    n_folds: int,
    ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[pd.DataFrame] = []
    subject_rows: list[dict[str, object]] = []

    for spec in model_specs:
        feature_cols = [
            col
            for col in frame.columns
            if any(col.startswith(prefix) for prefix in spec.prefixes)
        ]
        if not feature_cols:
            raise ValueError(f"No features matched model {spec.name!r}.")

        model_frame = frame.dropna(subset=feature_cols).copy()
        for subject, subject_df in model_frame.groupby("subject", sort=True):
            subject_df = subject_df.sort_values("trial_idx").reset_index(drop=True)
            if subject_df["label"].nunique() < 2:
                continue
            scores, y = decode_subject(
                subject_df,
                feature_cols=feature_cols,
                n_folds=n_folds,
                ridge=ridge,
            )
            scored = subject_df[
                ["subject", "trial_idx", "sdt", "confidence", "stimamp", "label", "decoder_fold"]
            ].copy()
            scored["model_name"] = spec.name
            scored["model_label"] = spec.label
            scored["decoder_score"] = scores
            score_rows.append(scored)

            auc = _roc_auc(y, scores)
            hit_scores = scores[y == 1]
            miss_scores = scores[y == 0]
            subject_rows.append(
                {
                    "model_name": spec.name,
                    "model_label": spec.label,
                    "subject": subject,
                    "n_trials": int(np.isfinite(scores).sum()),
                    "n_hits": int(np.sum(y == 1)),
                    "n_misses": int(np.sum(y == 0)),
                    "auc": auc,
                    "mean_hit_score": float(np.nanmean(hit_scores)) if hit_scores.size else math.nan,
                    "mean_miss_score": float(np.nanmean(miss_scores)) if miss_scores.size else math.nan,
                }
            )

    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(subject_rows)


def summarise_models(subject_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, model_label), model_df in subject_summary.groupby(["model_name", "model_label"], sort=False):
        auc = _finite(model_df["auc"])
        ci_low, ci_high = _ci95(auc)
        t = stats.ttest_1samp(auc, 0.5) if auc.size >= 2 else None
        rows.append(
            {
                "model_name": model_name,
                "model_label": model_label,
                "n_subjects": int(auc.size),
                "auc_mean": float(auc.mean()) if auc.size else math.nan,
                "auc_median": float(np.median(auc)) if auc.size else math.nan,
                "auc_sd": float(np.std(auc, ddof=1)) if auc.size > 1 else math.nan,
                "auc_ci95_low": ci_low,
                "auc_ci95_high": ci_high,
                "auc_above_half": int(np.sum(auc > 0.5)),
                "auc_ttest_vs_half_p": float(t.pvalue) if t is not None else math.nan,
            }
        )
    return pd.DataFrame(rows)


def summarise_pairwise(subject_summary: pd.DataFrame) -> pd.DataFrame:
    contrasts = (
        ("prism_gain_over_raw", "raw_central_delta", "raw_plus_prism", 0.0),
        ("var_gain_over_raw", "raw_central_delta", "raw_plus_var", 0.0),
        ("var_minus_prism_hybrid", "raw_plus_prism", "raw_plus_var", 0.0),
        ("prism_summary_minus_var_summary", "var_central_augmented", "prism_central_augmented", 0.0),
    )
    rows: list[dict[str, object]] = []
    for name, model_a, model_b, null_value in contrasts:
        wide = (
            subject_summary.loc[subject_summary["model_name"].isin([model_a, model_b])]
            .pivot(index="subject", columns="model_name", values="auc")
            .dropna()
        )
        if model_a not in wide.columns or model_b not in wide.columns:
            continue
        delta = wide[model_b] - wide[model_a]
        ci_low, ci_high = _ci95(delta)
        t = stats.ttest_1samp(delta.to_numpy(dtype=float), null_value) if delta.shape[0] >= 2 else None
        try:
            w_p = float(stats.wilcoxon(delta).pvalue)
        except ValueError:
            w_p = math.nan
        rows.append(
            {
                "contrast": name,
                "model_a": model_a,
                "model_b": model_b,
                "n_subjects": int(delta.shape[0]),
                "mean_auc_model_a": float(wide[model_a].mean()),
                "mean_auc_model_b": float(wide[model_b].mean()),
                "mean_delta_model_b_minus_a": float(delta.mean()),
                "median_delta_model_b_minus_a": float(delta.median()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_subjects": int(np.sum(delta > 0.0)),
                "ttest_p": float(t.pvalue) if t is not None else math.nan,
                "wilcoxon_p": w_p,
            }
        )
    return pd.DataFrame(rows)


def build_analysis_frame(
    *,
    baseline_results_dir: Path,
    prism_results_dir: Path,
    derivatives_dir: Path,
    export_dir: Path,
    region: str,
    rep_dim: int,
    baseline_rep_dim: int | None = None,
    test_start_ms: float,
    test_end_ms: float,
    baseline_start_ms: float,
    baseline_end_ms: float,
    prism_model_family: str,
    positive_sdt: str,
    negative_sdt: str,
    n_folds: int,
) -> pd.DataFrame:
    baseline_dim = int(rep_dim if baseline_rep_dim is None else baseline_rep_dim)
    var_features = load_baseline_features(
        baseline_results_dir,
        region=region,
        rep_dim=baseline_dim,
        test_start_ms=test_start_ms,
        test_end_ms=test_end_ms,
    )
    prism_features = load_prism_features(
        prism_results_dir,
        region=region,
        rep_dim=rep_dim,
        test_start_ms=test_start_ms,
        test_end_ms=test_end_ms,
        model_family=prism_model_family,
    )
    common_trials = (
        var_features[["subject", "trial_idx"]]
        .merge(prism_features[["subject", "trial_idx"]], on=["subject", "trial_idx"], how="inner")
        .drop_duplicates()
        .sort_values(["subject", "trial_idx"])
        .reset_index(drop=True)
    )
    if common_trials.empty:
        raise ValueError("No common trials were available for subject-wise decoding.")

    subjects = sorted(common_trials["subject"].astype(str).unique().tolist())
    metadata = load_metadata(
        subjects,
        derivatives_dir=derivatives_dir,
        positive_sdt=positive_sdt,
        negative_sdt=negative_sdt,
    )
    metadata = metadata.merge(common_trials, on=["subject", "trial_idx"], how="inner", validate="one_to_one")
    raw_features = build_raw_features(
        metadata[["subject", "trial_idx"]],
        export_dir=export_dir,
        derivatives_dir=derivatives_dir,
        region=region,
        baseline_start_ms=baseline_start_ms,
        baseline_end_ms=baseline_end_ms,
        test_start_ms=test_start_ms,
        test_end_ms=test_end_ms,
    )
    frame = (
        metadata[["subject", "trial_idx", "sdt", "confidence", "stimamp", "label"]]
        .merge(raw_features, on=["subject", "trial_idx"], how="inner", validate="one_to_one")
        .merge(var_features, on=["subject", "trial_idx"], how="inner", validate="one_to_one")
        .merge(prism_features, on=["subject", "trial_idx"], how="inner", validate="one_to_one")
    )
    frame = assign_subject_folds(frame, n_folds=n_folds)
    return frame


def plot_summary(subject_summary: pd.DataFrame, pairwise: pd.DataFrame, outdir: Path) -> None:
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism-mpl"))
        os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "prism-cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    order = [spec.name for spec in MODEL_SPECS]
    labels = {spec.name: spec.label for spec in MODEL_SPECS}
    wide = subject_summary.pivot(index="subject", columns="model_name", values="auc")
    x = np.arange(len(order), dtype=float)

    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    ax = axes[0]
    for _, row in wide.reindex(columns=order).iterrows():
        ax.plot(x, row.to_numpy(dtype=float), color="#c9c9c9", linewidth=0.8, zorder=1)
    palette = ["#4c6f91", "#7f7f7f", "#7b6aa7", "#b35c2e", "#6f9e75"]
    for idx, (model, colour) in enumerate(zip(order, palette)):
        values = wide[model].dropna().to_numpy(dtype=float)
        jitter = np.linspace(-0.07, 0.07, values.size) if values.size else np.asarray([])
        ax.scatter(np.full(values.size, idx) + jitter, values, s=16, color=colour, alpha=0.75, zorder=2)
        mean, low, high = values.mean(), *_ci95(values)
        ax.errorbar(idx, mean, yerr=[[mean - low], [high - mean]], fmt="o", color="#111111", capsize=3, zorder=3)
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--")
    short_labels = {
        "raw_central_delta": "Raw",
        "var_central_augmented": "VAR",
        "prism_central_augmented": "PRISM",
        "raw_plus_var": "Raw+VAR",
        "raw_plus_prism": "Raw+PRISM",
    }
    ax.set_xticks(x, [short_labels[model] for model in order], rotation=18, ha="right")
    ax.set_ylabel("Held-out AUC")
    ax.set_title("Within-subject decoder")
    ax.text(-0.11, 1.04, "A", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    focus = pairwise.loc[pairwise["contrast"].isin(["prism_gain_over_raw", "var_gain_over_raw", "var_minus_prism_hybrid"])]
    focus = focus.assign(
        label=focus["contrast"].map(
            {
                "prism_gain_over_raw": "PRISM gain\nvs raw",
                "var_gain_over_raw": "VAR gain\nvs raw",
                "var_minus_prism_hybrid": "VAR minus\nPRISM hybrid",
            }
        )
    )
    y_pos = np.arange(focus.shape[0], dtype=float)
    ax.barh(y_pos, focus["mean_delta_model_b_minus_a"], color=["#6f9e75", "#b35c2e", "#555555"][: focus.shape[0]])
    for idx, row in enumerate(focus.itertuples(index=False)):
        ax.errorbar(
            row.mean_delta_model_b_minus_a,
            idx,
            xerr=[[row.mean_delta_model_b_minus_a - row.ci95_low], [row.ci95_high - row.mean_delta_model_b_minus_a]],
            fmt="none",
            color="#111111",
            capsize=3,
            linewidth=1.0,
        )
    ax.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos, focus["label"])
    ax.set_xlabel("Delta AUC")
    ax.set_title("Subject-level contrasts")
    ax.text(-0.11, 1.04, "B", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.grid(axis="x", color="#eeeeee", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(outdir / "eeg_subject_decoder.png", dpi=300)
    fig.savefig(outdir / "eeg_subject_decoder.pdf")
    plt.close(fig)


def write_report(group_summary: pd.DataFrame, pairwise: pd.DataFrame, outpath: Path) -> None:
    lines = [
        "# Subject-Wise EEG Decoder",
        "",
        "Each decoder is trained and tested within a single subject using stratified folds.",
        "The group test then uses one AUC value per subject.",
        "",
        "## Model AUCs",
        "",
    ]
    for _, row in group_summary.iterrows():
        lines.append(
            f"- {row['model_name']}: mean AUC={row['auc_mean']:.4f}, "
            f"95% CI [{row['auc_ci95_low']:.4f}, {row['auc_ci95_high']:.4f}], "
            f"{int(row['auc_above_half'])}/{int(row['n_subjects'])} subjects above 0.5, "
            f"p={_format_p(float(row['auc_ttest_vs_half_p']))}."
        )
    lines.extend(["", "## Pairwise Contrasts", ""])
    for _, row in pairwise.iterrows():
        lines.append(
            f"- {row['contrast']}: delta={row['mean_delta_model_b_minus_a']:.4f}, "
            f"95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], "
            f"{int(row['positive_subjects'])}/{int(row['n_subjects'])} positive subjects, "
            f"paired t p={_format_p(float(row['ttest_p']))}, "
            f"Wilcoxon p={_format_p(float(row['wilcoxon_p']))}."
        )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    baseline_results_dir: Path,
    prism_results_dir: Path,
    derivatives_dir: Path,
    export_dir: Path,
    outdir: Path,
    region: str,
    rep_dim: int,
    baseline_rep_dim: int | None,
    test_start_ms: float,
    test_end_ms: float,
    baseline_start_ms: float,
    baseline_end_ms: float,
    prism_model_family: str,
    positive_sdt: str,
    negative_sdt: str,
    n_folds: int,
    ridge: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    frame = build_analysis_frame(
        baseline_results_dir=baseline_results_dir,
        prism_results_dir=prism_results_dir,
        derivatives_dir=derivatives_dir,
        export_dir=export_dir,
        region=region,
        rep_dim=rep_dim,
        baseline_rep_dim=baseline_rep_dim,
        test_start_ms=test_start_ms,
        test_end_ms=test_end_ms,
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

    frame.to_csv(outdir / "eeg_subject_decoder_features.csv", index=False)
    scores.to_csv(outdir / "eeg_subject_decoder_trial_scores.csv", index=False)
    subject_summary.to_csv(outdir / "eeg_subject_decoder_subject_summary.csv", index=False)
    group_summary.to_csv(outdir / "eeg_subject_decoder_group_summary.csv", index=False)
    pairwise.to_csv(outdir / "eeg_subject_decoder_pairwise.csv", index=False)
    write_report(group_summary, pairwise, outdir / "eeg_subject_decoder_summary.md")
    plot_summary(subject_summary, pairwise, outdir)
    print(f"Wrote {outdir / 'eeg_subject_decoder_group_summary.csv'}")
    print(f"Wrote {outdir / 'eeg_subject_decoder_pairwise.csv'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results-dir", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--prism-results-dir", type=Path, default=DEFAULT_PRISM_RESULTS)
    parser.add_argument("--derivatives-dir", type=Path, default=DEFAULT_DERIVATIVES_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--rep-dim", type=int, default=DEFAULT_REP_DIM)
    parser.add_argument("--baseline-rep-dim", type=int, default=None)
    parser.add_argument("--test-start-ms", type=float, default=DEFAULT_TEST_START_MS)
    parser.add_argument("--test-end-ms", type=float, default=DEFAULT_TEST_END_MS)
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
        baseline_results_dir=args.baseline_results_dir,
        prism_results_dir=args.prism_results_dir,
        derivatives_dir=args.derivatives_dir,
        export_dir=args.export_dir,
        outdir=args.outdir,
        region=args.region,
        rep_dim=args.rep_dim,
        baseline_rep_dim=args.baseline_rep_dim,
        test_start_ms=args.test_start_ms,
        test_end_ms=args.test_end_ms,
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
