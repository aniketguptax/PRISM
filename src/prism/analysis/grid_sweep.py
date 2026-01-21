import argparse
import json
import time
from pathlib import Path
from typing import List, cast

from prism.experiments.runner import run_experiment
from prism.experiments.registry import PROCESS_REGISTRY, RECONSTRUCTOR_REGISTRY
from prism.processes.wrappers import NoisyObservation, Subsample
from prism.representations import LastK
from prism.representations.protocols import Representation


def parse_int_list(xs: List[str]) -> List[int]:
    return [int(x) for x in xs]


def parse_float_list(xs: List[str]) -> List[float]:
    return [float(x) for x in xs]


def fmt_float(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--process", required=True, choices=PROCESS_REGISTRY.keys())
    parser.add_argument("--reconstructor", default="one_step", choices=RECONSTRUCTOR_REGISTRY.keys())

    parser.add_argument("--ks", nargs="+", required=True, help="k values, e.g. 1 2 3 4 5")
    parser.add_argument("--flip-ps", nargs="+", default=["0.0"], help="flip probabilities, e.g. 0.0 0.05 0.1")
    parser.add_argument("--subsample-steps", nargs="+", default=["1"], help="subsample steps, e.g. 1 2 4")

    parser.add_argument("--length", type=int, default=200_000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])

    parser.add_argument("--outdir", type=Path, default=Path("results/grid_sweep"))
    parser.add_argument("--save-transitions", action="store_true")
    parser.add_argument("--show-transitions-for", type=str, default=None)

    args = parser.parse_args()

    ks = sorted(set(parse_int_list(args.ks)))
    flip_ps = parse_float_list(args.flip_ps)
    subs = sorted(set(parse_int_list(args.subsample_steps)))

    reconstructor = RECONSTRUCTOR_REGISTRY[args.reconstructor]()
    reps = cast(List[Representation], [LastK(k=k) for k in ks])
    
    # Write top-level grid configuration for reproducibility
    args.outdir.mkdir(parents=True, exist_ok=True)

    grid_config = {
        "process": args.process,
        "reconstructor": args.reconstructor,
        "representations": [r.name for r in reps],
        "ks": ks,
        "flip_ps": flip_ps,
        "subsample_steps": subs,
        "length": args.length,
        "train_frac": args.train_frac,
        "seeds": args.seeds,
        "script": "grid_sweep.py",
        "created_at": int(time.time()),
    }

    config_path = args.outdir / "grid_config.json"
    
    if config_path.exists():
        raise FileExistsError(f"{config_path} already exists. Use a fresh --outdir to avoid mixing runs.")
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(grid_config, f, indent=2, sort_keys=True)

    if args.show_transitions_for is not None and not args.save_transitions:
        raise ValueError("Need --save-transitions when using --show-transitions-for")

    if args.show_transitions_for is not None:
        rep_names = {r.name for r in reps}
        if args.show_transitions_for not in rep_names:
            raise ValueError(f"--show-transitions-for {args.show_transitions_for} not in reps: {sorted(rep_names)}")
    
    for p in flip_ps:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"flip_p must be in [0,1], got {p}")

    for step in subs:
        if step < 1:
            raise ValueError(f"subsample_step must be >= 1, got {step}")

    for flip_p in flip_ps:
        for step in subs:
            proc = PROCESS_REGISTRY[args.process]()  # fresh instance per grid point
            tag_parts = [args.process]

            if flip_p > 0.0:
                proc = NoisyObservation(base=proc, flip_p=flip_p)
                tag_parts.append(f"flip{fmt_float(flip_p)}")

            if step > 1:
                proc = Subsample(base=proc, step=step)
                tag_parts.append(f"sub{step}")

            run_name = "_".join(tag_parts)
            out = args.outdir / run_name

            run_experiment(
                process=proc,
                reconstructor=reconstructor,
                representations=reps,
                length=args.length,
                train_frac=args.train_frac,
                seeds=args.seeds,
                outdir=out,
                save_transitions=args.save_transitions,
                transitions_rep_name=args.show_transitions_for if args.save_transitions else None,
            )

            print(f"wrote: {out/'metrics.csv'}")


if __name__ == "__main__":
    main()