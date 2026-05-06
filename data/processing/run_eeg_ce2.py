"""Fit ISS models per trial and save CE 2.0 label chains for EEG data."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eegprep import (
    DEFAULT_DERIVATIVES_DIR,
    load_and_prepare_subject,
    load_subject_channel_labels,
)
from experiments.models import fit_iss_label_chains
from experiments.spatial import select_region_groups
from framework.runtime import (
    add_all_subjects_argument,
    add_max_trials_argument,
    add_subject_source_arguments,
    extract_subject_id,
    resolve_subject_path,
    select_trial_indices,
)

WINDOWS = [
    ("train_-300_0_test_125_375",  -300.0,    0.0,  125.0, 375.0),
    ("train_-300_0_test_250_500",  -300.0,    0.0,  250.0, 500.0),
    ("train_-300_0_test_0_250",    -300.0,    0.0,    0.0, 250.0),
]

DEFAULT_REGIONS = ("central", "frontal", "parietal", "temporal", "occipital")
DEFAULT_EPS = (0.10, 0.15, 0.25, 0.40)
DEFAULT_REP_DIM = 4


def _process_subject(
    inpath: Path,
    outdir: Path,
    *,
    regions: tuple[str, ...],
    eps_values: tuple[float, ...],
    rep_dim: int,
    em_iters: int,
    em_tol: float,
    em_ridge: float,
    derivatives_dir: Path,
    max_trials: int | None,
) -> int:
    subject = extract_subject_id(inpath)
    data, _sfreq, times_ms = load_and_prepare_subject(inpath)
    n_trials = data.shape[0]

    channel_labels = load_subject_channel_labels(subject, derivatives_dir=derivatives_dir)
    try:
        selected_groups = select_region_groups(channel_labels, regions)
    except ValueError as exc:
        raise ValueError(f"{subject}: {exc}") from exc

    saved = 0
    rng = np.random.default_rng(sum(ord(c) for c in subject))
    trial_indices = select_trial_indices(n_trials, max_trials)

    for trial_idx in trial_indices:
        trial = data[trial_idx]  # (T, channels)

        for window_name, tr_start, tr_end, _te_start, _te_end in WINDOWS:
            mask_train = (times_ms >= tr_start) & (times_ms <= tr_end)
            train_window = trial[mask_train]  # (T_train, channels)

            for group in selected_groups:
                X_region = train_window[:, group.indices]

                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                chains = fit_iss_label_chains(
                    X_region,
                    rep_dim=min(rep_dim, X_region.shape[1]),
                    eps_values=eps_values,
                    em_iters=em_iters,
                    em_tol=em_tol,
                    em_ridge=em_ridge,
                    random_state=seed,
                )
                if chains is None:
                    continue

                arrays: dict[str, np.ndarray] = {
                    "subject":     np.array(subject),
                    "trial_idx":   np.array(trial_idx, dtype=int),
                    "region_name": np.array(group.name),
                    "window_name": np.array(window_name),
                }
                for eps, labels in chains.items():
                    key = f"labels_eps{eps:.4f}".rstrip("0").rstrip(".")
                    arrays[key] = labels.astype(np.int16)

                fname = (
                    outdir
                    / f"trial_{subject}_{group.name}_{window_name}_{trial_idx:05d}.npz"
                )
                np.savez_compressed(fname, **arrays)
                saved += 1

    print(f"  {subject}: saved {saved} trial NPZ files")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ISS CE 2.0 label extraction on EEG pre-stimulus windows."
    )
    add_subject_source_arguments(parser)
    add_all_subjects_argument(parser)
    add_max_trials_argument(parser)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./data/results_iss/eeg_ce2"),
        help="Directory for the per-trial CE 2.0 label archives.",
    )
    parser.add_argument("--rep-dim", type=int, default=DEFAULT_REP_DIM)
    parser.add_argument("--eps", type=float, nargs="+", default=list(DEFAULT_EPS))
    parser.add_argument(
        "--regions", nargs="+", default=list(DEFAULT_REGIONS),
        help="Anatomical region names to process.",
    )
    parser.add_argument("--em-iters", type=int, default=20)
    parser.add_argument("--em-tol", type=float, default=1e-3)
    parser.add_argument("--em-ridge", type=float, default=1e-6)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--derivatives-dir",
        default=str(DEFAULT_DERIVATIVES_DIR),
        help="Path to EEGLAB derivatives (needed for channel labels).",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    export_dir = Path(args.export_dir)

    if args.all_subjects:
        paths = sorted(export_dir.glob("*_export.mat"))
    else:
        paths = [resolve_subject_path(args.subject, args.input_path, export_dir)]

    kwargs = dict(
        outdir=args.outdir,
        regions=tuple(args.regions),
        eps_values=tuple(float(e) for e in args.eps),
        rep_dim=args.rep_dim,
        em_iters=args.em_iters,
        em_tol=args.em_tol,
        em_ridge=args.em_ridge,
        derivatives_dir=Path(args.derivatives_dir),
        max_trials=args.max_trials,
    )

    print(f"Processing {len(paths)} subject(s) into {args.outdir}")

    if args.n_jobs > 1 and len(paths) > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futs = {pool.submit(_process_subject, p, **kwargs): p for p in paths}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:
                    print(f"  ERROR for {futs[fut].name}: {exc}", file=sys.stderr)
    else:
        for p in paths:
            _process_subject(p, **kwargs)


if __name__ == "__main__":
    main()
