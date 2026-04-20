"""Behaviour-linked metadata loading and summary helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from eegprep import resolve_preproc_set_path
from framework.summaries import flatten_columns, save_table


DEFAULT_DERIVATIVES_DIR = Path("./data/ds001785/derivatives/eegprep")
BASELINE_REQUIRED_COLUMNS = [
    "subject",
    "trial_idx",
    "rep_dim",
    "pred_mse_obs",
    "pred_r2_obs",
]


def _coerce_scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        if value.size == 1:
            return _coerce_scalar(value.item())
        return value.tolist()
    return value


def _safe_float(value) -> float:
    value = _coerce_scalar(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value) -> int:
    value = _coerce_scalar(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _safe_str(value) -> str:
    value = _coerce_scalar(value)
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def load_subject_epoch_metadata(
    subject: str,
    derivatives_dir: Path = DEFAULT_DERIVATIVES_DIR,
) -> pd.DataFrame:
    set_path = resolve_preproc_set_path(subject, derivatives_dir=derivatives_dir)
    if not set_path.exists():
        raise FileNotFoundError(f"Preprocessed EEGLAB file not found: {set_path}")

    eeg = loadmat(set_path, struct_as_record=False, squeeze_me=True)["EEG"]
    epochs = np.atleast_1d(eeg.epoch)

    rows = []
    for trial_idx, epoch in enumerate(epochs):
        rows.append(
            {
                "subject": subject,
                "trial_idx": int(trial_idx),
                "epoch_id": _safe_int(getattr(epoch, "id", None)),
                "task_type": _safe_str(getattr(epoch, "type", None)),
                "sdt": _safe_str(getattr(epoch, "sdt", None)),
                "confidence": _safe_float(getattr(epoch, "conf", None)),
                "stimamp": _safe_float(getattr(epoch, "stimamp", None)),
                "stimonset": _safe_float(getattr(epoch, "stimonset", None)),
                "rt": _safe_float(getattr(epoch, "rt", None)),
                "rt_conf": _safe_float(getattr(epoch, "rt_conf", None)),
                "first_rpeak": _safe_float(getattr(epoch, "first_rpeak", None)),
                "last_rpeak": _safe_float(getattr(epoch, "last_rpeak", None)),
            }
        )

    return pd.DataFrame(rows)


def load_all_epoch_metadata(
    subjects: list[str],
    derivatives_dir: Path = DEFAULT_DERIVATIVES_DIR,
) -> pd.DataFrame:
    frames = [
        load_subject_epoch_metadata(subject, derivatives_dir=derivatives_dir)
        for subject in subjects
    ]
    return pd.concat(frames, ignore_index=True)


def load_baseline_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [col for col in BASELINE_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Baseline CSV is missing required columns: {missing}")
    return df


def compute_trial_capacity_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (subject, trial_idx), trial_df in merged_df.groupby(["subject", "trial_idx"]):
        trial_df = trial_df.sort_values("rep_dim").reset_index(drop=True)
        best_idx = int(trial_df["pred_r2_obs"].idxmax())
        best_q = int(trial_df.loc[best_idx, "rep_dim"])
        best_r2 = float(trial_df.loc[best_idx, "pred_r2_obs"])
        best_mse = float(trial_df["pred_mse_obs"].min())

        if best_r2 > 0.0:
            target_r2 = 0.95 * best_r2
            q_eff95_rows = trial_df.loc[trial_df["pred_r2_obs"] >= target_r2]
            q_eff95_r2 = int(q_eff95_rows.iloc[0]["rep_dim"]) if not q_eff95_rows.empty else best_q
        else:
            q_eff95_r2 = best_q

        target_mse = 1.05 * best_mse
        q_eff105_mse = int(
            trial_df.loc[trial_df["pred_mse_obs"] <= target_mse].iloc[0]["rep_dim"]
        )

        base_row = trial_df.iloc[0]
        rows.append(
            {
                "subject": subject,
                "trial_idx": int(trial_idx),
                "epoch_id": int(base_row["epoch_id"]),
                "task_type": base_row["task_type"],
                "sdt": base_row["sdt"],
                "confidence": float(base_row["confidence"]),
                "stimamp": float(base_row["stimamp"]),
                "stimonset": float(base_row["stimonset"]),
                "rt": float(base_row["rt"]),
                "rt_conf": float(base_row["rt_conf"]),
                "best_q": best_q,
                "best_r2_obs": best_r2,
                "best_mse_obs": best_mse,
                "q_eff95_r2": q_eff95_r2,
                "q_eff105_mse": q_eff105_mse,
            }
        )

    return pd.DataFrame(rows)


def summarise_by_sdt(trial_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_level = (
        trial_summary.groupby("sdt")[["best_r2_obs", "best_mse_obs", "confidence"]]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    trial_level = flatten_columns(trial_level)

    subject_level = (
        trial_summary.groupby(["subject", "sdt"])[["best_r2_obs", "best_mse_obs", "confidence"]]
        .mean()
        .reset_index()
    )

    group_level = (
        subject_level.groupby("sdt")[["best_r2_obs", "best_mse_obs", "confidence"]]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    return trial_level, flatten_columns(group_level)


def summarise_q16_by_sdt(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q16 = merged_df.loc[merged_df["rep_dim"] == 16].copy()

    trial_level = (
        q16.groupby("sdt")[["pred_r2_obs", "pred_mse_obs", "confidence"]]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    trial_level = flatten_columns(trial_level)

    subject_level = (
        q16.groupby(["subject", "sdt"])[["pred_r2_obs", "pred_mse_obs", "confidence"]]
        .mean()
        .reset_index()
    )

    group_level = (
        subject_level.groupby("sdt")[["pred_r2_obs", "pred_mse_obs", "confidence"]]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    return trial_level, flatten_columns(group_level)


def summarise_confidence_bins(
    trial_summary: pd.DataFrame,
    n_bins: int = 4,
) -> pd.DataFrame:
    df = trial_summary.dropna(subset=["confidence"]).copy()
    df["confidence_bin"] = pd.qcut(df["confidence"], n_bins, duplicates="drop")
    summary = (
        df.groupby("confidence_bin", observed=False)[
            ["best_r2_obs", "best_mse_obs", "q_eff95_r2", "q_eff105_mse"]
        ]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    return flatten_columns(summary)


def run_behaviour_summary(
    *,
    baseline_csv: Path,
    derivatives_dir: Path,
    outdir: Path,
) -> int:
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        baseline_df = load_baseline_results(baseline_csv)
        subjects = sorted(baseline_df["subject"].unique().tolist())
        metadata_df = load_all_epoch_metadata(subjects, derivatives_dir=derivatives_dir)
        merged_df = baseline_df.merge(
            metadata_df,
            on=["subject", "trial_idx"],
            how="left",
            validate="many_to_one",
        )
        trial_summary = compute_trial_capacity_summary(merged_df)
        capacity_by_sdt_trial, capacity_by_sdt_subject = summarise_by_sdt(trial_summary)
        q16_by_sdt_trial, q16_by_sdt_subject = summarise_q16_by_sdt(merged_df)
        confidence_summary = summarise_confidence_bins(trial_summary)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    merged_path = outdir / "all_subjects_trial_baseline_with_metadata.csv"
    trial_summary_path = outdir / "all_subjects_trial_capacity_summary.csv"
    capacity_sdt_trial_path = outdir / "all_subjects_capacity_by_sdt_trial_level.csv"
    capacity_sdt_subject_path = outdir / "all_subjects_capacity_by_sdt_subject_level.csv"
    q16_sdt_trial_path = outdir / "all_subjects_q16_by_sdt_trial_level.csv"
    q16_sdt_subject_path = outdir / "all_subjects_q16_by_sdt_subject_level.csv"
    confidence_path = outdir / "all_subjects_capacity_by_confidence_bin.csv"

    save_table(merged_df, merged_path)
    save_table(trial_summary, trial_summary_path)
    save_table(capacity_by_sdt_trial, capacity_sdt_trial_path)
    save_table(capacity_by_sdt_subject, capacity_sdt_subject_path)
    save_table(q16_by_sdt_trial, q16_sdt_trial_path)
    save_table(q16_by_sdt_subject, q16_sdt_subject_path)
    save_table(confidence_summary, confidence_path)

    print(f"Saved {merged_path}")
    print(f"Saved {trial_summary_path}")
    print(f"Saved {capacity_sdt_trial_path}")
    print(f"Saved {capacity_sdt_subject_path}")
    print(f"Saved {q16_sdt_trial_path}")
    print(f"Saved {q16_sdt_subject_path}")
    print(f"Saved {confidence_path}")

    print("\nSubject-level q=16 summary by SDT outcome:")
    print(q16_by_sdt_subject.to_string(index=False))

    print("\nCapacity summary by confidence bin:")
    print(confidence_summary.to_string(index=False))
    return 0
