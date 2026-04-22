"""Unified summary command for EEG processing results."""

from __future__ import annotations

import argparse

from reporting.registry import SUMMARY_TASKS, get_summary_task


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task_name", required=True)

    for spec in SUMMARY_TASKS:
        subparser = subparsers.add_parser(
            spec.name,
            aliases=list(spec.aliases),
            help=spec.description,
            description=spec.description,
        )
        spec.add_arguments(subparser)
        subparser.set_defaults(task_name=spec.name)

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    spec = get_summary_task(args.task_name)
    return spec.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
