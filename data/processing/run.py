"""Unified runner for EEG processing experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import eegprep
from eegprep import load_and_prepare_subject
from experiments.registry import EXPERIMENTS, get_experiment
from framework.runtime import (
    add_all_subjects_argument,
    add_max_trials_argument,
    add_n_jobs_argument,
    add_outdir_argument,
    add_rep_dims_argument,
    add_subject_source_arguments,
    add_train_fraction_argument,
    count_error_rows,
    extract_subject_id,
    resolve_subject_path,
    run_all_subjects_batch,
    run_subject_trial_evaluation,
)


def build_inspect_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "inspect",
        help="Check one exported subject file and print a small loader summary.",
        description="Check one exported subject file and print a small loader summary.",
    )
    add_subject_source_arguments(parser)
    parser.set_defaults(command_name="inspect")


def build_experiment_parser(subparsers, spec) -> None:
    parser = subparsers.add_parser(
        spec.name,
        aliases=list(spec.aliases),
        help=spec.description,
        description=spec.description,
    )
    add_subject_source_arguments(parser)
    add_outdir_argument(parser, default_outdir=spec.default_outdir)
    add_all_subjects_argument(parser)
    if spec.train_fraction_help is not None:
        add_train_fraction_argument(parser, help_text=spec.train_fraction_help)
    if spec.rep_dims_default is not None:
        add_rep_dims_argument(
            parser,
            default_rep_dims=spec.rep_dims_default,
            help_text=spec.rep_dims_help,
        )
    if spec.add_arguments is not None:
        spec.add_arguments(parser)
    add_max_trials_argument(parser)
    add_n_jobs_argument(parser)
    parser.set_defaults(command_name=spec.name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    build_inspect_parser(subparsers)
    for spec in EXPERIMENTS:
        build_experiment_parser(subparsers, spec)
    return parser


def inspect_subject(args: argparse.Namespace) -> int:
    path = resolve_subject_path(args.subject, args.input_path, Path(args.export_dir))

    try:
        data, sfreq, times = load_and_prepare_subject(path)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    print(eegprep.__file__)
    print("data shape:", data.shape)
    print("sfreq:", sfreq)
    print("times shape:", times.shape)

    X0 = data[0]
    print("trial 0 shape:", X0.shape)
    print("mean over time, first 5 channels:", X0.mean(axis=0)[:5])
    print("std over time, first 5 channels:", X0.std(axis=0)[:5])
    print("any NaN:", bool(np.isnan(data).any()))
    print("any inf:", bool(np.isinf(data).any()))
    print("global mean:", float(data.mean()))
    print("global std:", float(data.std()))
    return 0


def run_subject_file(spec, *, inpath: Path, outdir: Path, config: dict, max_trials: int | None):
    evaluate_trial_fn = spec.evaluate_trial_factory(config, inpath)
    return run_subject_trial_evaluation(
        inpath=inpath,
        outdir=outdir,
        evaluate_trial_fn=evaluate_trial_fn,
        output_columns=list(spec.output_columns),
        output_suffix=spec.output_suffix,
        load_subject_fn=spec.load_subject_fn,
        time_bounds_keys=spec.time_bounds_keys,
        progress_every=spec.progress_every,
        max_trials=max_trials,
    )


def run_single_subject(spec, args: argparse.Namespace) -> int:
    config = {} if spec.config_from_args is None else spec.config_from_args(args)
    inpath = resolve_subject_path(args.subject, args.input_path, Path(args.export_dir))
    outdir = Path(args.outdir)

    try:
        df, outfile = run_subject_file(
            spec,
            inpath=inpath,
            outdir=outdir,
            config=config,
            max_trials=args.max_trials,
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    subject = extract_subject_id(inpath)
    if spec.describe_run is None:
        print(f"Processed {df['trial_idx'].nunique()} trials for {subject}")
    else:
        print(spec.describe_run(df, subject))

    print(f"Saved {len(df)} rows to {outfile}")
    n_errors = count_error_rows(df)
    if n_errors:
        print(f"Rows with errors: {n_errors}")
    return 0


def run_all_subjects(spec, args: argparse.Namespace) -> int:
    config = {} if spec.config_from_args is None else spec.config_from_args(args)

    def subject_runner(*, inpath: Path, outdir: Path, **_kwargs):
        return run_subject_file(
            spec,
            inpath=inpath,
            outdir=outdir,
            config=config,
            max_trials=args.max_trials,
        )

    try:
        run_all_subjects_batch(
            export_dir=Path(args.export_dir),
            outdir=Path(args.outdir),
            run_subject_file_fn=subject_runner,
            combined_filename=spec.combined_filename,
            n_jobs=args.n_jobs,
            sort_columns=spec.sort_columns,
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.command_name == "inspect":
        return inspect_subject(args)

    spec = get_experiment(args.command_name)
    if args.all_subjects:
        return run_all_subjects(spec, args)
    return run_single_subject(spec, args)


if __name__ == "__main__":
    raise SystemExit(main())
