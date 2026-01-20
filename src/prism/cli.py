import argparse
import json
from pathlib import Path
from typing import List

from prism.experiments.runner import run_experiment
from prism.experiments.registry import PROCESS_REGISTRY, RECONSTRUCTOR_REGISTRY
from prism.representations import LastK
from prism.representations.discrete import LastKWithNoise
from prism.representations.protocols import Representation
from prism.utils.rng import bernoulli_noise


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--process", required=True, choices=PROCESS_REGISTRY.keys())
    parser.add_argument("--reconstructor", default="one_step", choices=RECONSTRUCTOR_REGISTRY.keys())
    
    parser.add_argument("--ks", nargs="+", type=int, required=True, help="k values for LastK")
    parser.add_argument("--length", type=int, default=400_000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="random seeds")
    
    parser.add_argument("--noisy", action="store_true", help="add noise to representation")
    parser.add_argument("--noise-seed", type=int, default=123, help="noise seed")
    
    parser.add_argument("--outdir", type=Path, default=Path("results/run"))
    parser.add_argument("--save-transitions", action="store_true", help="save state transitions")
    parser.add_argument(
        "--show-transitions-for",
        type=str,
        default=None,
        help="which rep to show transitions for"
    )
    parser.add_argument(
        "--show-seed",
        type=int,
        default=None,
        help="seed for showing transitions"
    )

    args = parser.parse_args()

    process = PROCESS_REGISTRY[args.process]()
    reconstructor = RECONSTRUCTOR_REGISTRY[args.reconstructor]()

    reps: List[Representation] = []
    if args.noisy:
        noise = bernoulli_noise(length=args.length, seed=args.noise_seed)
        reps = [LastKWithNoise(k=k, noise=noise) for k in args.ks]
    else:
        reps = [LastK(k=k) for k in args.ks]

    run_experiment(
        process=process,
        reconstructor=reconstructor,
        representations=reps,
        length=args.length,
        train_frac=args.train_frac,
        seeds=args.seeds,
        outdir=args.outdir,
        save_transitions=args.save_transitions,
        transitions_rep_name=args.show_transitions_for if args.save_transitions else None,
    )
    
    if args.show_transitions_for is not None:
        if not args.save_transitions:
            print("\nneed --save-transitions to show transitions")
            return

        seed_to_show = args.show_seed if args.show_seed is not None else args.seeds[0]
        transitions_file = args.outdir / f"transitions_{args.show_transitions_for}_seed{seed_to_show}.json"

        if transitions_file.exists():
            edges = json.loads(transitions_file.read_text(encoding="utf-8"))
            print(f"\n{args.show_transitions_for} (seed={seed_to_show}):")
            for s, sym, sp, p in sorted(edges, key=lambda e: (e[0], e[1], e[2])):
                print(f"S{s} --{sym}: {p:.3f}--> S{sp}")
            dot_file = args.outdir / f"{args.process}_{args.show_transitions_for}_seed{seed_to_show}.dot"
            print(f"\nwrote {dot_file}")
        else:
            print(f"\nno transitions file at {transitions_file}")


if __name__ == "__main__":
    main()