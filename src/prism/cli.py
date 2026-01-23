import argparse
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from prism.experiments.runner import run_experiment
from prism.experiments.registry import PROCESS_REGISTRY, RECONSTRUCTOR_REGISTRY
from prism.representations import LastK
from prism.representations.discrete import LastKWithNoise
from prism.representations.protocols import Representation
from prism.processes.wrappers import NoisyObservation, Subsample
from prism.utils.rng import bernoulli_noise
from prism.utils.io import save_json

def _make_outdir(base: Path, process_name: str, reconstructor_name: str,
                 length: int, seeds: List[int], run_id: Optional[str]) -> Path:
    if base != Path("results/run"):
        return base
    timestamp = datetime.utcnow().strftime(("%Y-%m-%d_%H%M%S"))
    s0, s1 = min(seeds), max(seeds)
    r_id = run_id or f"{timestamp}_{process_name}__{reconstructor_name}__L{length}__s{s0}-{s1}"
    return Path("results") / r_id

def _validate_sweep(flip_ps: List[float], subsample_steps: List[int]) -> None:
    for flip_p in flip_ps:
        if not (0.0 <= flip_p <= 1.0):
            raise ValueError(f"Invalid flip probability: {flip_p}")
    for subsample_step in subsample_steps:
        if subsample_step < 1:
            raise ValueError(f"Invalid subsample step: {subsample_step}")

def _condition_dict(process_name: str, flip_p: float, subsample_step: int, cond_id_style: str) -> dict:
    wrappers = []
    if flip_p > 0.0:
        wrappers.append(f"flip={flip_p:g}")
    if subsample_step > 1:
        wrappers.append(f"sub={subsample_step}")
        
    if cond_id_style == "json":
        cond_id = json.dumps({
            "flip_p": flip_p,
            "subsample_step": subsample_step
        }, sort_keys=True)
    else:
        cond_id = f"flip{flip_p:g}_sub{subsample_step}"
    return {
        "base_process": process_name,
        "flip_p": flip_p,
        "subsample_step": subsample_step,
        "wrappers": "|".join(wrappers),
        "condition_id": cond_id
    }


def main():
    parser = argparse.ArgumentParser()
    
    # core experiment settings
    parser.add_argument("--process", required=True, choices=PROCESS_REGISTRY.keys())
    parser.add_argument("--reconstructor", default="one_step", choices=RECONSTRUCTOR_REGISTRY.keys())
    
    # representation settings
    parser.add_argument("--ks", nargs="+", type=int, required=True, help="k values for LastK")
    parser.add_argument("--length", type=int, default=400_000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="random seeds")
    parser.add_argument("--eps", type=float, default=0.02, help="Epsilon for OneStepGreedyMerge")
    
    # output settings
    parser.add_argument("--noisy", action="store_true", help="add noise to representation")
    parser.add_argument("--noise-seed", type=int, default=123, help="noise seed")
    parser.add_argument("--outdir", type=Path, default=Path("results/run"))
    parser.add_argument("--force", action="store_true", help="allow writing into an existing run directory")
    parser.add_argument("--run-id", type=str, default=None, help="optional run ID to distinguish runs")
    
    # sweep settings
    parser.add_argument("--flip-p", type=float, default=0.0, help="bit-flip probability on observations")
    parser.add_argument("--subsample-step", type=int, default=1, help="keep every k-th sample")
    parser.add_argument("--flip-ps", nargs="+", type=float, default=None, help="Sweep of flip probabilities. If set, overrides --flip-p.")
    parser.add_argument("--subsample-steps", nargs="+", type=int, default=None, help="Sweep of subsample steps. If set, overrides --subsample-step.")
    parser.add_argument("--condition-id-style", type=str, default="flip_sub",
                        choices=["flip_sub", "json"],
                        help="How to encode condition_id/wrappers fields.")
    
    # transitions
    parser.add_argument("--save-transitions", action="store_true", help="save state transitions")
    parser.add_argument("--show-transitions-for", type=str, default=None, help="which rep to show transitions for")
    parser.add_argument("--show-seed", type=int, default=None, help="seed for showing transitions" )
    parser.add_argument("--show-flip-p", type=float, default=None, help="which flip_p to show transitions for (sweeps)")
    parser.add_argument("--show-subsample-step", type=int, default=None, help="which subsample_step to show transitions for (sweeps)")

    args = parser.parse_args()
    
    flip_ps = args.flip_ps if args.flip_ps is not None else [args.flip_p]
    subsample_steps = args.subsample_steps if args.subsample_steps is not None else [args.subsample_step]
    _validate_sweep(flip_ps, subsample_steps)
    
    args.outdir = _make_outdir(
        base=args.outdir,
        process_name=args.process,
        reconstructor_name=args.reconstructor,
        length=args.length,
        seeds=args.seeds,
        run_id=args.run_id
    )
    
    runs_csv = args.outdir / "runs.csv"
    if runs_csv.exists():
        if not args.force:
            raise FileExistsError(f"Output directory {args.outdir} already exists (use a fresh --outdir, or pass --force to overwrite)")
        # overwrite existing runs.csv
        runs_csv.unlink()

    # prepare representations
    reps: List[Representation] = []
    if args.noisy:
        noise = bernoulli_noise(length=args.length, seed=args.noise_seed)
        reps = [LastKWithNoise(k=k, noise=noise) for k in args.ks]
    else:
        reps = [LastK(k=k) for k in args.ks]
    
    if args.eps is None:
        reconstructor = RECONSTRUCTOR_REGISTRY[args.reconstructor]()
    else:
        reconstructor = RECONSTRUCTOR_REGISTRY[args.reconstructor](eps=args.eps)
    
    # write sweep-level config
    config = {
        "process": args.process,
        "reconstructor": args.reconstructor,
        "eps": getattr(reconstructor, "eps", None),
        "length": args.length,
        "train_frac": args.train_frac,
        "seeds": args.seeds,
        "representations": [r.name for r in reps],
        "flip_ps": flip_ps,
        "subsample_steps": subsample_steps,
        "noisy_representation": args.noisy,
        "noise_seed": args.noise_seed if args.noisy else None,
        "save_transitions": args.save_transitions,
        "transitions_rep_name": args.show_transitions_for if args.save_transitions else None,
        "condition_id_style": args.condition_id_style,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    save_json(args.outdir / "config.json", config)
    
    # run sweep
    for flip_p in flip_ps:
        for step in subsample_steps:
            process = PROCESS_REGISTRY[args.process]()
            if flip_p > 0.0:
                process = NoisyObservation(base=process, flip_p=flip_p)
            
            if step > 1:
                process = Subsample(base=process, step=step)
            
            condition = _condition_dict(
                process_name=args.process,
                flip_p=flip_p,
                subsample_step=step,
                cond_id_style=args.condition_id_style
            )
            
            run_experiment(
                process=process,
                reconstructor=reconstructor,
                representations=reps,
                length=args.length,
                train_frac=args.train_frac,
                seeds=args.seeds,
                outdir=args.outdir,
                condition=condition,
                save_transitions=args.save_transitions,
                transitions_rep_name=args.show_transitions_for if args.save_transitions else None,
            )
    
    # show transitions if requested
    if args.show_transitions_for is not None:
        if not args.save_transitions:
            print("\nneed --save-transitions to show transitions")
            return

        seed_to_show = args.show_seed if args.show_seed is not None else args.seeds[0]
        show_flip_p = args.show_flip_p if args.show_flip_p is not None else flip_ps[0]
        show_subsample_step = args.show_subsample_step if args.show_subsample_step is not None else subsample_steps[0]
        
        cond = _condition_dict(
            process_name=args.process,
            flip_p=show_flip_p,
            subsample_step=show_subsample_step,
            cond_id_style=args.condition_id_style
        )
        cond_id = cond["condition_id"]
        
        transitions_file = (
            args.outdir 
            / "transitions" 
            / f"{cond_id}__transitions_{args.show_transitions_for}_seed{seed_to_show}.json"
        )

        if transitions_file.exists():
            edges = json.loads(transitions_file.read_text(encoding="utf-8"))
            print(f"\n{args.show_transitions_for} (seed={seed_to_show}, condition_id={cond_id}):")
            for s, sym, sp, p in sorted(edges, key=lambda e: (e[0], e[1], e[2])):
                print(f"S{s} --{sym}: {p:.3f}--> S{sp}")
            dot_file = (
                args.outdir 
                / "transitions" 
                / f"{cond_id}__transitions_{args.show_transitions_for}_seed{seed_to_show}.dot"
            )
            print(f"\nwrote {dot_file}")
        else:
            print(f"\nno transitions file at {transitions_file}")


if __name__ == "__main__":
    main()