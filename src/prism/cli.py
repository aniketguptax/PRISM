"""Command-line entry point for PRISM experiments."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from prism.experiments.registry import PROCESS_REGISTRY, RECONSTRUCTOR_REGISTRY
from prism.experiments.runner import run_experiment
from prism.processes.wrappers import NoisyObservation, Subsample
from prism.representations import ISSDim, LastK
from prism.representations.discrete import LastKWithNoise
from prism.representations.protocols import Representation
from prism.utils.io import save_json
from prism.utils.rng import bernoulli_noise

CONTINUOUS_PROCESSES = {"continuous_file", "linear_gaussian_ssm"}


def _make_outdir(
    base: Path,
    process_name: str,
    reconstructor_name: str,
    length: int,
    seeds: Sequence[int],
    run_id: Optional[str],
) -> Path:
    if base != Path("results/run"):
        return base
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    seed_min, seed_max = min(seeds), max(seeds)
    generated_id = run_id or f"{timestamp}_{process_name}__{reconstructor_name}__L{length}__s{seed_min}-{seed_max}"
    return Path("results") / generated_id


def _validate_sweep(flip_ps: Sequence[float], subsample_steps: Sequence[int]) -> None:
    for flip_p in flip_ps:
        if not (0.0 <= flip_p <= 1.0):
            raise ValueError(f"Invalid flip probability: {flip_p}")
    for subsample_step in subsample_steps:
        if subsample_step < 1:
            raise ValueError(f"Invalid subsample step: {subsample_step}")


def _condition_dict(process_name: str, flip_p: float, subsample_step: int, cond_id_style: str) -> dict[str, object]:
    wrappers: list[str] = []
    if flip_p > 0.0:
        wrappers.append(f"flip={flip_p:g}")
    if subsample_step > 1:
        wrappers.append(f"sub={subsample_step}")

    if cond_id_style == "json":
        condition_id = json.dumps(
            {"flip_p": flip_p, "subsample_step": subsample_step},
            sort_keys=True,
        )
    else:
        condition_id = f"flip{flip_p:g}_sub{subsample_step}"

    return {
        "base_process": process_name,
        "flip_p": flip_p,
        "subsample_step": subsample_step,
        "wrappers": "|".join(wrappers),
        "condition_id": condition_id,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--process", required=True, choices=PROCESS_REGISTRY.keys())
    parser.add_argument("--reconstructor", default="one_step", choices=RECONSTRUCTOR_REGISTRY.keys())

    parser.add_argument("--ks", nargs="+", type=int, required=True, help="k values (discrete) or d values (ISS)")
    parser.add_argument("--length", type=int, default=400_000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds.")

    parser.add_argument("--eps", type=float, default=0.02, help="Epsilon for one-step merge.")
    parser.add_argument("--em-iters", type=int, default=50, help="EM iterations for Kalman ISS.")
    parser.add_argument("--em-tol", type=float, default=1e-4, help="EM convergence tolerance.")
    parser.add_argument("--em-ridge", type=float, default=1e-6, help="Ridge stabilizer for ISS EM.")

    parser.add_argument("--data-path", type=Path, default=None, help="Path to .npy/.csv/.txt time series.")
    parser.add_argument("--data-column", type=int, default=None, help="Column index for multi-column files.")

    parser.add_argument("--lgssm-a", type=float, default=0.92, help="AR coefficient for linear_gaussian_ssm.")
    parser.add_argument("--lgssm-c", type=float, default=1.0, help="Observation loading for linear_gaussian_ssm.")
    parser.add_argument("--lgssm-process-std", type=float, default=0.35, help="Process noise std.")
    parser.add_argument("--lgssm-obs-std", type=float, default=0.25, help="Observation noise std.")
    parser.add_argument("--lgssm-init-std", type=float, default=1.0, help="Initial latent std.")

    parser.add_argument("--noisy", action="store_true", help="Add deterministic noise bit to representation.")
    parser.add_argument("--noise-seed", type=int, default=123, help="Noise seed used by LastKWithNoise.")

    parser.add_argument("--outdir", type=Path, default=Path("results/run"))
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing run directory.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run ID.")

    parser.add_argument("--flip-p", type=float, default=0.0, help="Bit flip probability on observations.")
    parser.add_argument("--subsample-step", type=int, default=1, help="Keep every k-th sample.")
    parser.add_argument("--flip-ps", nargs="+", type=float, default=None, help="Sweep of flip probabilities.")
    parser.add_argument("--subsample-steps", nargs="+", type=int, default=None, help="Sweep of subsample steps.")
    parser.add_argument(
        "--condition-id-style",
        type=str,
        default="flip_sub",
        choices=["flip_sub", "json"],
        help="How to encode condition_id/wrappers fields.",
    )

    parser.add_argument("--save-transitions", action="store_true", help="Save discrete state transitions.")
    parser.add_argument("--show-transitions-for", type=str, default=None, help="Representation name to print.")
    parser.add_argument("--show-seed", type=int, default=None, help="Seed for transition inspection.")
    parser.add_argument("--show-flip-p", type=float, default=None, help="flip_p for transition inspection.")
    parser.add_argument("--show-subsample-step", type=int, default=None, help="subsample step for transition inspection.")

    return parser.parse_args()


def _build_process(args: argparse.Namespace):
    if args.process == "continuous_file":
        if args.data_path is None:
            raise ValueError("--data-path is required for --process continuous_file")
        return PROCESS_REGISTRY[args.process](path=args.data_path, column=args.data_column)
    if args.process == "linear_gaussian_ssm":
        return PROCESS_REGISTRY[args.process](
            a=args.lgssm_a,
            c=args.lgssm_c,
            process_std=args.lgssm_process_std,
            obs_std=args.lgssm_obs_std,
            init_std=args.lgssm_init_std,
        )
    return PROCESS_REGISTRY[args.process]()


def _build_representations(args: argparse.Namespace) -> list[Representation]:
    if args.reconstructor == "kalman_iss":
        return [ISSDim(d=k) for k in args.ks]
    if args.noisy:
        noise = bernoulli_noise(length=args.length, seed=args.noise_seed)
        return [LastKWithNoise(k=k, noise=noise) for k in args.ks]
    return [LastK(k=k) for k in args.ks]


def main() -> None:
    args = _parse_args()
    flip_ps = args.flip_ps if args.flip_ps is not None else [args.flip_p]
    subsample_steps = args.subsample_steps if args.subsample_steps is not None else [args.subsample_step]
    _validate_sweep(flip_ps, subsample_steps)

    if args.reconstructor == "kalman_iss" and args.process not in CONTINUOUS_PROCESSES:
        raise ValueError(
            f"--reconstructor kalman_iss requires a continuous process. "
            f"Choose one of: {sorted(CONTINUOUS_PROCESSES)}."
        )
    if args.reconstructor == "one_step" and args.process in CONTINUOUS_PROCESSES:
        raise ValueError("--reconstructor one_step only supports discrete binary processes.")
    if args.reconstructor == "kalman_iss" and args.noisy:
        raise ValueError("--noisy is only valid for discrete LastK representations.")
    if args.reconstructor == "kalman_iss" and args.save_transitions:
        raise ValueError("Continuous Kalman ISS runs do not support --save-transitions.")
    if args.reconstructor == "kalman_iss" and args.show_transitions_for is not None:
        raise ValueError("Continuous Kalman ISS runs do not support --show-transitions-for.")

    args.outdir = _make_outdir(
        base=args.outdir,
        process_name=args.process,
        reconstructor_name=args.reconstructor,
        length=args.length,
        seeds=args.seeds,
        run_id=args.run_id,
    )
    runs_csv = args.outdir / "runs.csv"
    if runs_csv.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory {args.outdir} already exists. Use --force or a fresh --outdir."
            )
        runs_csv.unlink()

    representations = _build_representations(args)
    if args.reconstructor == "kalman_iss":
        reconstructor = RECONSTRUCTOR_REGISTRY[args.reconstructor](
            em_iters=args.em_iters,
            em_tol=args.em_tol,
            em_ridge=args.em_ridge,
        )
    else:
        reconstructor = RECONSTRUCTOR_REGISTRY[args.reconstructor](eps=args.eps)

    config = {
        "process": args.process,
        "reconstructor": args.reconstructor,
        "eps": getattr(reconstructor, "eps", None),
        "length": args.length,
        "train_frac": args.train_frac,
        "seeds": args.seeds,
        "representations": [r.name for r in representations],
        "flip_ps": flip_ps,
        "subsample_steps": subsample_steps,
        "noisy_representation": args.noisy,
        "noise_seed": args.noise_seed if args.noisy else None,
        "save_transitions": args.save_transitions,
        "transitions_rep_name": args.show_transitions_for if args.save_transitions else None,
        "condition_id_style": args.condition_id_style,
        "data_path": str(args.data_path) if args.data_path is not None else None,
        "data_column": args.data_column,
        "kalman_iss": {
            "em_iters": args.em_iters,
            "em_tol": args.em_tol,
            "em_ridge": args.em_ridge,
        }
        if args.reconstructor == "kalman_iss"
        else None,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    save_json(args.outdir / "config.json", config)

    for flip_p in flip_ps:
        for step in subsample_steps:
            process = _build_process(args)
            if flip_p > 0.0:
                if args.process in CONTINUOUS_PROCESSES:
                    raise ValueError(
                        "Observation flip noise is a binary wrapper and cannot be applied to continuous processes."
                    )
                process = NoisyObservation(base=process, flip_p=flip_p)
            if step > 1:
                process = Subsample(base=process, step=step)

            condition = _condition_dict(
                process_name=args.process,
                flip_p=flip_p,
                subsample_step=step,
                cond_id_style=args.condition_id_style,
            )
            run_experiment(
                process=process,
                reconstructor=reconstructor,
                representations=representations,
                length=args.length,
                train_frac=args.train_frac,
                seeds=args.seeds,
                outdir=args.outdir,
                condition=condition,
                save_transitions=args.save_transitions,
                transitions_rep_name=args.show_transitions_for if args.save_transitions else None,
            )

    if args.show_transitions_for is None:
        return
    if not args.save_transitions:
        raise ValueError("--show-transitions-for requires --save-transitions.")

    seed_to_show = args.show_seed if args.show_seed is not None else args.seeds[0]
    show_flip_p = args.show_flip_p if args.show_flip_p is not None else flip_ps[0]
    show_subsample = args.show_subsample_step if args.show_subsample_step is not None else subsample_steps[0]
    cond = _condition_dict(
        process_name=args.process,
        flip_p=show_flip_p,
        subsample_step=show_subsample,
        cond_id_style=args.condition_id_style,
    )
    cond_id = cond["condition_id"]
    transitions_file = args.outdir / "transitions" / f"{cond_id}__transitions_{args.show_transitions_for}_seed{seed_to_show}.json"
    if not transitions_file.exists():
        print(f"\nNo transitions file at {transitions_file}")
        return

    edges = json.loads(transitions_file.read_text(encoding="utf-8"))
    print(f"\n{args.show_transitions_for} (seed={seed_to_show}, condition_id={cond_id}):")
    for s, symbol, sp, prob in sorted(edges, key=lambda e: (e[0], e[1], e[2])):
        print(f"S{s} --{symbol}: {prob:.3f}--> S{sp}")
    dot_file = args.outdir / "transitions" / f"{cond_id}__transitions_{args.show_transitions_for}_seed{seed_to_show}.dot"
    print(f"\nWrote {dot_file}")


if __name__ == "__main__":
    main()
