"""Shared runtime helpers for the EEG processing commands."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from eegprep import load_exported_subject


EXPORT_GLOB = "*_export.mat"


def extract_subject_id(path: Path) -> str:
    return path.name.split("_")[0]


def resolve_subject_path(
    subject: str | None,
    input_path: str | None,
    export_dir: Path,
) -> Path:
    if input_path is not None:
        return Path(input_path)
    if subject is None:
        raise ValueError("Either --subject or --input-path must be provided")
    return export_dir / f"{subject}_preproc_01hz_export.mat"


def add_subject_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--subject",
        default="sub-01",
        help="Subject id such as sub-01. Ignored if --input-path is given.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Direct path to one exported MATLAB file.",
    )
    parser.add_argument(
        "--export-dir",
        default="./data/exports_mat",
        help="Directory containing exported MATLAB subject files.",
    )


def add_outdir_argument(
    parser: argparse.ArgumentParser,
    default_outdir: str,
    help_text: str = "Directory for the CSV outputs.",
) -> None:
    parser.add_argument(
        "--outdir",
        default=default_outdir,
        help=help_text,
    )


def add_all_subjects_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Process every exported subject file rather than just one subject.",
    )


def add_train_fraction_argument(
    parser: argparse.ArgumentParser,
    help_text: str,
    default: float = 0.7,
) -> None:
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=default,
        help=help_text,
    )


def add_rep_dims_argument(
    parser: argparse.ArgumentParser,
    default_rep_dims: tuple[int, ...],
    help_text: str = "Representation dimensions to evaluate.",
) -> None:
    parser.add_argument(
        "--rep-dims",
        nargs="+",
        type=int,
        default=list(default_rep_dims),
        help=help_text,
    )


def add_n_jobs_argument(parser: argparse.ArgumentParser, default: int = 1) -> None:
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=default,
        help="Number of subjects to process in parallel with threads.",
    )


def add_max_trials_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="If set, run an evenly spaced deterministic subset of trials per subject.",
    )


def load_subject_export(inpath: Path):
    if not inpath.exists():
        raise FileNotFoundError(f"Input file not found: {inpath}")

    data, sfreq, times = load_exported_subject(str(inpath))
    if data.ndim != 3:
        raise ValueError(
            f"Expected subject data with shape (trials, time, channels), got {data.shape}"
        )
    if times.ndim != 1:
        raise ValueError(f"Expected 1D times array, got {times.shape}")
    if data.shape[1] != times.shape[0]:
        raise ValueError(
            f"Time axis mismatch: data has {data.shape[1]} samples, times has {times.shape[0]}"
        )

    return data, float(sfreq), times


def finalise_results_frame(results: list[dict], output_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    for column in output_columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df.loc[:, output_columns]


def count_error_rows(df: pd.DataFrame, error_column: str = "error") -> int:
    if error_column not in df.columns:
        return 0
    return int(df[error_column].fillna("").astype(str).ne("").sum())


def select_trial_indices(n_trials: int, max_trials: int | None) -> list[int]:
    if max_trials is None or max_trials >= n_trials:
        return list(range(n_trials))
    if max_trials < 1:
        raise ValueError("--max-trials must be at least 1 when provided")
    if max_trials == 1:
        return [0]

    return np.linspace(0, n_trials - 1, num=max_trials, dtype=int).tolist()


def run_subject_trial_evaluation(
    inpath: Path,
    outdir: Path,
    evaluate_trial_fn: Callable,
    output_columns: list[str],
    output_suffix: str,
    *,
    load_subject_fn: Callable = load_subject_export,
    time_bounds_keys: tuple[str, str] = ("trial_tmin_ms", "trial_tmax_ms"),
    progress_every: int = 100,
    max_trials: int | None = None,
) -> tuple[pd.DataFrame, Path]:
    data, sfreq, times = load_subject_fn(inpath)

    outdir.mkdir(parents=True, exist_ok=True)
    subject = extract_subject_id(inpath)
    shared_row_data = {
        "sfreq": float(sfreq),
        time_bounds_keys[0]: float(times[0]),
        time_bounds_keys[1]: float(times[-1]),
    }
    results: list[dict] = []
    trial_indices = select_trial_indices(data.shape[0], max_trials)
    n_selected = len(trial_indices)

    for loop_idx, trial_idx in enumerate(trial_indices, start=1):
        trial_rows = evaluate_trial_fn(data[trial_idx], times)
        for row in trial_rows:
            row["subject"] = subject
            row["trial_idx"] = int(trial_idx)
            row.update(shared_row_data)

        results.extend(trial_rows)

        if progress_every > 0 and loop_idx % progress_every == 0:
            print(f"{subject}: processed {loop_idx}/{n_selected} selected trials")

    df = finalise_results_frame(results, output_columns)
    outfile = outdir / f"{subject}{output_suffix}"
    df.to_csv(outfile, index=False)
    return df, outfile


def list_export_files(export_dir: Path) -> list[Path]:
    if not export_dir.exists():
        raise FileNotFoundError(f"export directory not found: {export_dir}")

    files = sorted(export_dir.glob(EXPORT_GLOB))
    if not files:
        raise FileNotFoundError(f"no exported subject files found in {export_dir}")

    return files


def run_all_subjects_batch(
    export_dir: Path,
    outdir: Path,
    run_subject_file_fn: Callable,
    combined_filename: str,
    *,
    subject_run_kwargs: dict | None = None,
    n_jobs: int = 1,
    sort_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, Path, list[tuple[str, str]]]:
    files = list_export_files(export_dir)
    if n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")

    outdir.mkdir(parents=True, exist_ok=True)
    subject_run_kwargs = {} if subject_run_kwargs is None else dict(subject_run_kwargs)
    combined_frames: list[pd.DataFrame] = []
    failed_subjects: list[tuple[str, str]] = []

    print(f"Found {len(files)} exported subjects")

    if n_jobs == 1:
        for inpath in files:
            try:
                df, outfile = run_subject_file_fn(
                    inpath=inpath,
                    outdir=outdir,
                    **subject_run_kwargs,
                )
                combined_frames.append(df)
                print(f"Saved {outfile}")
            except Exception as exc:
                failed_subjects.append((inpath.name, str(exc)))
                print(f"FAILED on {inpath.name}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_subject = {
                executor.submit(
                    run_subject_file_fn,
                    inpath=inpath,
                    outdir=outdir,
                    **subject_run_kwargs,
                ): inpath.name
                for inpath in files
            }
            for future in as_completed(future_to_subject):
                subject_name = future_to_subject[future]
                try:
                    df, outfile = future.result()
                    combined_frames.append(df)
                    print(f"Saved {outfile}")
                except Exception as exc:
                    failed_subjects.append((subject_name, str(exc)))
                    print(f"FAILED on {subject_name}: {exc}")

    if not combined_frames:
        raise RuntimeError("no subject results were written")

    combined = pd.concat(combined_frames, ignore_index=True)
    if sort_columns is not None:
        available = [column for column in sort_columns if column in combined.columns]
        if available:
            combined = combined.sort_values(available)

    combined_out = outdir / combined_filename
    combined.to_csv(combined_out, index=False)

    print(f"Saved combined results to {combined_out}")
    if "subject" in combined.columns:
        print(combined.groupby("subject").size())

    if failed_subjects:
        print("Subjects that failed completely:")
        for subject_name, error in failed_subjects:
            print(f"  {subject_name}: {error}")

    return combined, combined_out, failed_subjects
