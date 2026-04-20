"""Stimulus-amplitude-stratified evidence summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from framework.summaries import prepare_pyplot, save_table
from reporting.behaviour import DEFAULT_DERIVATIVES_DIR, load_all_epoch_metadata
from reporting.evidence import DEFAULT_MODEL_ORDER, DEFAULT_POSITIVE_SDT, roc_auc_score_binary


DEFAULT_TEMPORAL_TRIAL_SCORES_CSV = Path(
    "./data/results_baseline/region_sliding_baseline300ms_controls_focus_q4/"
    "summary_temporal_evidence_check/temporal_evidence_trial_scores.csv"
)
DEFAULT_MODEL_NAMES = (
    "raw_regions_delta",
    "baseline_regions_augmented",
    "raw_plus_baseline_regions_augmented",
)
DEFAULT_STIMAMP_BIN_LABELS = ("low", "mid", "high")
DEFAULT_FOCUS_MODEL_A = "raw_regions_delta"
DEFAULT_FOCUS_MODEL_B = "raw_plus_baseline_regions_augmented"
DEFAULT_FOCUS_START_MS = 125.0
DEFAULT_FOCUS_END_MS = 375.0


def build_model_order(present_models: list[str]) -> list[str]:
    preferred = [*DEFAULT_MODEL_NAMES, *DEFAULT_MODEL_ORDER]
    ordered = [name for name in preferred if name in present_models]
    ordered.extend(name for name in present_models if name not in ordered)
    return ordered


def load_temporal_trial_scores(trial_scores_csv: Path) -> pd.DataFrame:
    if not trial_scores_csv.exists():
        raise FileNotFoundError(f"Temporal evidence trial-score CSV not found: {trial_scores_csv}")

    df = pd.read_csv(trial_scores_csv)
    required = {
        "subject",
        "trial_idx",
        "sdt",
        "confidence",
        "label",
        "model_name",
        "model_label",
        "evidence_score",
        "target_start_ms",
        "target_end_ms",
        "target_center_ms",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Temporal evidence trial-score CSV is missing required columns: {missing}")
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


def summarise_bin_counts(trial_scores: pd.DataFrame) -> pd.DataFrame:
    unique_trials = (
        trial_scores[["subject", "trial_idx", "stimamp_bin", "sdt"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    summary = (
        unique_trials.groupby("stimamp_bin", observed=False)
        .agg(
            n_trials=("trial_idx", "size"),
            n_subjects=("subject", "nunique"),
            n_hits=("sdt", lambda s: int(np.sum(s == "hit"))),
            n_misses=("sdt", lambda s: int(np.sum(s == "miss"))),
        )
        .reset_index()
    )
    return summary


def assign_within_subject_stimamp_bins(
    trial_metadata: pd.DataFrame,
    *,
    n_bins: int,
    bin_labels: tuple[str, ...],
) -> pd.DataFrame:
    if n_bins <= 1:
        raise ValueError("n_bins must be greater than one")

    if len(bin_labels) != n_bins:
        raise ValueError("bin_labels must match n_bins")

    binned = trial_metadata.copy()
    binned["stimamp_bin"] = ""

    for subject, subject_df in binned.groupby("subject", observed=False):
        idx = subject_df.index
        values = subject_df["stimamp"].astype(float)
        if values.notna().sum() < n_bins:
            raise ValueError(f"Not enough valid stimamp values to form {n_bins} bins for {subject}")

        ranked = values.rank(method="first")
        codes = pd.qcut(ranked, q=n_bins, labels=False)
        labels = pd.Series(codes, index=subject_df.index).map(lambda code: bin_labels[int(code)])
        binned.loc[idx, "stimamp_bin"] = labels.to_numpy(dtype=object)

    return binned


def build_stimamp_trial_table(
    trial_scores: pd.DataFrame,
    *,
    derivatives_dir: Path,
    n_bins: int,
    bin_labels: tuple[str, ...],
) -> pd.DataFrame:
    subjects = sorted(trial_scores["subject"].dropna().astype(str).unique().tolist())
    metadata = load_stimamp_metadata(subjects, derivatives_dir=derivatives_dir)

    trial_metadata = (
        trial_scores[["subject", "trial_idx"]]
        .drop_duplicates()
        .merge(
            metadata,
            on=["subject", "trial_idx"],
            how="left",
            validate="one_to_one",
        )
    )
    missing = trial_metadata["stimamp"].isna()
    if missing.any():
        missing_rows = trial_metadata.loc[missing, ["subject", "trial_idx"]]
        raise ValueError(
            "Stimamp metadata was missing for some trials, for example "
            f"{missing_rows.iloc[0].to_dict()}"
        )

    trial_metadata = assign_within_subject_stimamp_bins(
        trial_metadata,
        n_bins=n_bins,
        bin_labels=bin_labels,
    )
    merged = trial_scores.merge(
        trial_metadata,
        on=["subject", "trial_idx"],
        how="inner",
        validate="many_to_one",
    )
    return merged.sort_values(
        ["target_start_ms", "model_name", "subject", "trial_idx"]
    ).reset_index(drop=True)


def evaluate_stimamp_subject_metrics(
    trial_scores: pd.DataFrame,
    *,
    positive_sdt: str,
) -> pd.DataFrame:
    rows = []
    group_columns = [
        "target_start_ms",
        "target_end_ms",
        "target_center_ms",
        "stimamp_bin",
        "model_name",
        "model_label",
        "subject",
    ]
    for keys, subject_df in trial_scores.groupby(group_columns, observed=False):
        (
            target_start_ms,
            target_end_ms,
            target_center_ms,
            stimamp_bin,
            model_name,
            model_label,
            subject,
        ) = keys
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
                "target_start_ms": float(target_start_ms),
                "target_end_ms": float(target_end_ms),
                "target_center_ms": float(target_center_ms),
                "stimamp_bin": str(stimamp_bin),
                "model_name": str(model_name),
                "model_label": str(model_label),
                "subject": str(subject),
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


def summarise_stimamp_metrics(
    trial_scores: pd.DataFrame,
    subject_summary: pd.DataFrame,
    *,
    positive_sdt: str,
    bin_labels: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    group_columns = [
        "target_start_ms",
        "target_end_ms",
        "target_center_ms",
        "stimamp_bin",
        "model_name",
        "model_label",
    ]
    for keys, group_df in subject_summary.groupby(group_columns, observed=False):
        (
            target_start_ms,
            target_end_ms,
            target_center_ms,
            stimamp_bin,
            model_name,
            model_label,
        ) = keys
        auc_values = group_df["auc"].dropna().to_numpy(dtype=float)
        rho_values = group_df["hit_conf_rho"].dropna().to_numpy(dtype=float)
        pooled_df = trial_scores.loc[
            (trial_scores["target_start_ms"] == float(target_start_ms))
            & (trial_scores["target_end_ms"] == float(target_end_ms))
            & (trial_scores["stimamp_bin"] == str(stimamp_bin))
            & (trial_scores["model_name"] == str(model_name))
        ].copy()
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

        auc_t = stats.ttest_1samp(auc_values, 0.5, nan_policy="omit") if auc_values.size >= 2 else None
        rho_t = stats.ttest_1samp(rho_values, 0.0, nan_policy="omit") if rho_values.size >= 2 else None
        rows.append(
            {
                "target_start_ms": float(target_start_ms),
                "target_end_ms": float(target_end_ms),
                "target_center_ms": float(target_center_ms),
                "stimamp_bin": str(stimamp_bin),
                "model_name": str(model_name),
                "model_label": str(model_label),
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
    ordered_names = build_model_order(present_names)
    out["model_name"] = pd.Categorical(out["model_name"], categories=ordered_names, ordered=True)
    out["stimamp_bin"] = pd.Categorical(out["stimamp_bin"], categories=list(bin_labels), ordered=True)
    return out.sort_values(["target_start_ms", "stimamp_bin", "model_name"]).reset_index(drop=True)


def build_stimamp_pairwise_comparisons(
    subject_summary: pd.DataFrame,
    *,
    bin_labels: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    window_columns = ["target_start_ms", "target_end_ms", "target_center_ms", "stimamp_bin"]
    for keys, window_df in subject_summary.groupby(window_columns, observed=False):
        target_start_ms, target_end_ms, target_center_ms, stimamp_bin = keys
        present_models = window_df["model_name"].dropna().astype(str).unique().tolist()
        models = build_model_order(present_models)
        for model_a_idx, model_a in enumerate(models):
            for model_b in models[model_a_idx + 1 :]:
                wide = (
                    window_df.loc[window_df["model_name"].isin([model_a, model_b])]
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
                            "target_start_ms": float(target_start_ms),
                            "target_end_ms": float(target_end_ms),
                            "target_center_ms": float(target_center_ms),
                            "stimamp_bin": str(stimamp_bin),
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

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["stimamp_bin"] = pd.Categorical(out["stimamp_bin"], categories=list(bin_labels), ordered=True)
    return out.sort_values(["target_start_ms", "stimamp_bin", "metric", "model_a", "model_b"]).reset_index(
        drop=True
    )


def select_focus_pair_rows(
    pairwise_summary: pd.DataFrame,
    *,
    focus_model_a: str,
    focus_model_b: str,
    metric: str = "auc",
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
    pair_df.loc[reverse, "mean_delta_model_b_minus_a"] = -pair_df.loc[reverse, "mean_delta_model_b_minus_a"]
    pair_df.loc[reverse, "ttest_stat"] = -pair_df.loc[reverse, "ttest_stat"]
    pair_df.loc[reverse, "positive_subject_share"] = float("nan")
    pair_df["contrast_label"] = f"{focus_model_b} minus {focus_model_a}"
    return pair_df.sort_values(["target_start_ms", "stimamp_bin"]).reset_index(drop=True)


def choose_focus_window(
    pairwise_summary: pd.DataFrame,
    *,
    focus_model_a: str,
    focus_model_b: str,
    default_start_ms: float,
    default_end_ms: float,
) -> tuple[float, float]:
    focus_pairs = select_focus_pair_rows(
        pairwise_summary,
        focus_model_a=focus_model_a,
        focus_model_b=focus_model_b,
    )
    exact = focus_pairs.loc[
        (focus_pairs["target_start_ms"] == float(default_start_ms))
        & (focus_pairs["target_end_ms"] == float(default_end_ms))
    ]
    if not exact.empty:
        return float(default_start_ms), float(default_end_ms)
    if focus_pairs.empty:
        raise ValueError("No pairwise rows were available for the requested focus models")

    mean_delta = (
        focus_pairs.groupby(["target_start_ms", "target_end_ms"], observed=False)["mean_delta_model_b_minus_a"]
        .mean()
        .reset_index()
        .sort_values("mean_delta_model_b_minus_a", ascending=False)
    )
    best = mean_delta.iloc[0]
    return float(best["target_start_ms"]), float(best["target_end_ms"])


def plot_stimamp_evidence_summary(
    group_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    *,
    model_names: tuple[str, ...],
    focus_model_a: str,
    focus_model_b: str,
    bin_labels: tuple[str, ...],
    outfile: Path,
) -> None:
    if group_summary.empty:
        return

    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, len(bin_labels) + 1, figsize=(17, 4.6), constrained_layout=True)
    palette = {
        "raw_regions_delta": "#1d3557",
        "baseline_regions_augmented": "#457b9d",
        "raw_plus_baseline_regions_augmented": "#e76f51",
    }

    for ax, stimamp_bin in zip(axes[:-1], bin_labels):
        bin_df = group_summary.loc[group_summary["stimamp_bin"] == stimamp_bin].copy()
        for model_name in model_names:
            model_df = bin_df.loc[bin_df["model_name"] == model_name].sort_values("target_start_ms")
            if model_df.empty:
                continue
            ax.plot(
                model_df["target_start_ms"],
                model_df["auc_mean"],
                marker="o",
                linewidth=2.2,
                color=palette.get(model_name, "#495057"),
                label=model_df["model_label"].iloc[0],
            )
        ax.axhline(0.5, color="#6c757d", linestyle="--", linewidth=1.0)
        ax.set_title(f"{stimamp_bin.title()} stimamp")
        ax.set_xlabel("Target window start (ms)")
        ax.set_ylabel("Subject mean AUC")
        ax.grid(True, alpha=0.25)

    delta_ax = axes[-1]
    focus_pairs = select_focus_pair_rows(
        pairwise_summary,
        focus_model_a=focus_model_a,
        focus_model_b=focus_model_b,
    )
    bin_palette = {
        "low": "#2a9d8f",
        "mid": "#8d99ae",
        "high": "#bc4749",
    }
    for stimamp_bin in bin_labels:
        bin_df = focus_pairs.loc[focus_pairs["stimamp_bin"] == stimamp_bin].sort_values("target_start_ms")
        if bin_df.empty:
            continue
        delta_ax.plot(
            bin_df["target_start_ms"],
            bin_df["mean_delta_model_b_minus_a"],
            marker="o",
            linewidth=2.2,
            color=bin_palette.get(stimamp_bin, "#495057"),
            label=stimamp_bin.title(),
        )
    delta_ax.axhline(0.0, color="#6c757d", linestyle="--", linewidth=1.0)
    delta_ax.set_title("Hybrid minus raw AUC")
    delta_ax.set_xlabel("Target window start (ms)")
    delta_ax.set_ylabel(f"{focus_model_b} - {focus_model_a}")
    delta_ax.grid(True, alpha=0.25)
    delta_ax.legend(frameon=False, loc="best")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(3, len(labels)))

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_stimamp_evidence_report(
    group_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    *,
    focus_model_a: str,
    focus_model_b: str,
    focus_start_ms: float,
    focus_end_ms: float,
    bin_labels: tuple[str, ...],
) -> str:
    lines = [
        "# Stimamp Evidence Summary",
        "",
        (
            "Stimamp bins are within-subject quantile bins. "
            "That keeps the comparison on a relative per-participant scale, which matters here because "
            "the vibrotactile amplitude was adjusted around threshold."
        ),
        "",
    ]

    focus_pairs = select_focus_pair_rows(
        pairwise_summary,
        focus_model_a=focus_model_a,
        focus_model_b=focus_model_b,
    )
    if focus_pairs.empty:
        lines.append("No valid pairwise comparisons were available.")
        return "\n".join(lines)

    focus_window_df = focus_pairs.loc[
        (focus_pairs["target_start_ms"] == float(focus_start_ms))
        & (focus_pairs["target_end_ms"] == float(focus_end_ms))
    ].copy()
    if not focus_window_df.empty:
        lines.extend(
            [
                f"## {int(focus_start_ms)} to {int(focus_end_ms)} ms",
                "",
            ]
        )
        for stimamp_bin in bin_labels:
            row_df = focus_window_df.loc[focus_window_df["stimamp_bin"] == stimamp_bin]
            if row_df.empty:
                continue
            row = row_df.iloc[0]
            lines.append(
                f"- {stimamp_bin}: {focus_model_b} minus {focus_model_a} delta AUC = "
                f"{row['mean_delta_model_b_minus_a']:.3f} "
                f"(p={row['ttest_p']:.3g}, positive-subject share={row['positive_subject_share']:.3f})."
            )
        lines.append("")

        baseline_vs_raw = pairwise_summary.loc[
            (pairwise_summary["metric"] == "auc")
            & (pairwise_summary["target_start_ms"] == float(focus_start_ms))
            & (pairwise_summary["target_end_ms"] == float(focus_end_ms))
            & (pairwise_summary["model_a"] == "raw_regions_delta")
            & (pairwise_summary["model_b"] == "baseline_regions_augmented")
        ].copy()
        hybrid_vs_baseline = pairwise_summary.loc[
            (pairwise_summary["metric"] == "auc")
            & (pairwise_summary["target_start_ms"] == float(focus_start_ms))
            & (pairwise_summary["target_end_ms"] == float(focus_end_ms))
            & (pairwise_summary["model_a"] == "baseline_regions_augmented")
            & (pairwise_summary["model_b"] == "raw_plus_baseline_regions_augmented")
        ].copy()
        if not baseline_vs_raw.empty or not hybrid_vs_baseline.empty:
            lines.extend(["## Interpretation at the focus window", ""])
            for stimamp_bin in bin_labels:
                raw_row = baseline_vs_raw.loc[baseline_vs_raw["stimamp_bin"] == stimamp_bin]
                hybrid_row = hybrid_vs_baseline.loc[hybrid_vs_baseline["stimamp_bin"] == stimamp_bin]
                if raw_row.empty or hybrid_row.empty:
                    continue
                raw_row = raw_row.iloc[0]
                hybrid_row = hybrid_row.iloc[0]
                lines.append(
                    f"- {stimamp_bin}: baseline-only minus raw delta AUC = "
                    f"{raw_row['mean_delta_model_b_minus_a']:.3f} (p={raw_row['ttest_p']:.3g}), "
                    f"while hybrid minus baseline-only delta AUC = "
                    f"{hybrid_row['mean_delta_model_b_minus_a']:.3f} (p={hybrid_row['ttest_p']:.3g})."
                )
            lines.append("")

    lines.extend(["## Temporal pattern", ""])
    for stimamp_bin in bin_labels:
        bin_df = focus_pairs.loc[focus_pairs["stimamp_bin"] == stimamp_bin].sort_values(
            "mean_delta_model_b_minus_a",
            ascending=False,
        )
        if bin_df.empty:
            continue
        best = bin_df.iloc[0]
        lines.append(
            f"- {stimamp_bin}: strongest hybrid gain was {int(best['target_start_ms'])} to "
            f"{int(best['target_end_ms'])} ms with delta AUC {best['mean_delta_model_b_minus_a']:.3f} "
            f"(p={best['ttest_p']:.3g})."
        )

    return "\n".join(lines)


def run_stimamp_evidence_summary(
    *,
    trial_scores_csv: Path = DEFAULT_TEMPORAL_TRIAL_SCORES_CSV,
    derivatives_dir: Path = DEFAULT_DERIVATIVES_DIR,
    outdir: Path | None = None,
    model_names: tuple[str, ...] = DEFAULT_MODEL_NAMES,
    positive_sdt: str = DEFAULT_POSITIVE_SDT,
    n_bins: int = 3,
    bin_labels: tuple[str, ...] = DEFAULT_STIMAMP_BIN_LABELS,
    focus_model_a: str = DEFAULT_FOCUS_MODEL_A,
    focus_model_b: str = DEFAULT_FOCUS_MODEL_B,
    focus_start_ms: float = DEFAULT_FOCUS_START_MS,
    focus_end_ms: float = DEFAULT_FOCUS_END_MS,
) -> int:
    outdir = (
        trial_scores_csv.parent / "summary_stimamp_evidence"
        if outdir is None
        else outdir
    )
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        trial_scores = load_temporal_trial_scores(trial_scores_csv)
        trial_scores = filter_models(trial_scores, model_names=model_names)
        stimamp_trial_scores = build_stimamp_trial_table(
            trial_scores,
            derivatives_dir=derivatives_dir,
            n_bins=n_bins,
            bin_labels=bin_labels,
        )
        subject_summary = evaluate_stimamp_subject_metrics(
            stimamp_trial_scores,
            positive_sdt=positive_sdt,
        )
        group_summary = summarise_stimamp_metrics(
            stimamp_trial_scores,
            subject_summary,
            positive_sdt=positive_sdt,
            bin_labels=bin_labels,
        )
        pairwise_summary = build_stimamp_pairwise_comparisons(
            subject_summary,
            bin_labels=bin_labels,
        )
        focus_start_ms, focus_end_ms = choose_focus_window(
            pairwise_summary,
            focus_model_a=focus_model_a,
            focus_model_b=focus_model_b,
            default_start_ms=focus_start_ms,
            default_end_ms=focus_end_ms,
        )
        report_text = build_stimamp_evidence_report(
            group_summary,
            pairwise_summary,
            focus_model_a=focus_model_a,
            focus_model_b=focus_model_b,
            focus_start_ms=focus_start_ms,
            focus_end_ms=focus_end_ms,
            bin_labels=bin_labels,
        )
        bin_counts = summarise_bin_counts(stimamp_trial_scores)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    trial_scores_path = outdir / "stimamp_evidence_trial_scores.csv"
    subject_summary_path = outdir / "stimamp_evidence_subject_summary.csv"
    group_summary_path = outdir / "stimamp_evidence_group_summary.csv"
    pairwise_summary_path = outdir / "stimamp_evidence_pairwise_summary.csv"
    bin_counts_path = outdir / "stimamp_evidence_bin_counts.csv"
    report_path = outdir / "stimamp_evidence_summary.md"
    plot_path = outdir / "stimamp_evidence_auc.png"

    save_table(stimamp_trial_scores, trial_scores_path)
    save_table(subject_summary, subject_summary_path)
    save_table(group_summary, group_summary_path)
    save_table(pairwise_summary, pairwise_summary_path)
    save_table(bin_counts, bin_counts_path)
    report_path.write_text(report_text)
    plot_stimamp_evidence_summary(
        group_summary,
        pairwise_summary,
        model_names=model_names,
        focus_model_a=focus_model_a,
        focus_model_b=focus_model_b,
        bin_labels=bin_labels,
        outfile=plot_path,
    )

    print(f"Saved {trial_scores_path}")
    print(f"Saved {subject_summary_path}")
    print(f"Saved {group_summary_path}")
    print(f"Saved {pairwise_summary_path}")
    print(f"Saved {bin_counts_path}")
    print(f"Saved {report_path}")
    print(f"Saved {plot_path}")

    print("\nFocus window:")
    print(f"  {focus_start_ms:.0f} to {focus_end_ms:.0f} ms")

    focus_pairs = select_focus_pair_rows(
        pairwise_summary,
        focus_model_a=focus_model_a,
        focus_model_b=focus_model_b,
    )
    focus_window_df = focus_pairs.loc[
        (focus_pairs["target_start_ms"] == float(focus_start_ms))
        & (focus_pairs["target_end_ms"] == float(focus_end_ms))
    ].copy()
    if not focus_window_df.empty:
        print("\nHybrid minus raw by stimamp bin:")
        print(
            focus_window_df[
                [
                    "stimamp_bin",
                    "n_subjects",
                    "mean_model_a",
                    "mean_model_b",
                    "mean_delta_model_b_minus_a",
                    "positive_subject_share",
                    "ttest_p",
                ]
            ].to_string(index=False)
        )

    return 0
