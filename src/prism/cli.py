import argparse
import json
import logging
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional, Sequence

from prism.experiments.registry import PROCESS_REGISTRY, RECONSTRUCTOR_REGISTRY
from prism.experiments.runner import run_experiment
from prism.processes.wrappers import NoisyObservation, Subsample
from prism.representations import ISSDim, LastK
from prism.representations.discrete import LastKWithNoise
from prism.representations.protocols import Representation
from prism.utils.io import save_json
from prism.utils.logging import configure_logging
from prism.utils.rng import bernoulli_noise

CONTINUOUS_PROCESSES = {"continuous_file", "linear_gaussian_ssm"}
LOGGER = logging.getLogger(__name__)


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
    parser.add_argument("--dvs", nargs="+", type=int, default=[1], help="Macro projection dimensions d_V (continuous ISS).")
    parser.add_argument("--length", type=int, default=400_000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds.")

    parser.add_argument("--eps", type=float, default=0.02, help="Epsilon for one-step merge.")
    parser.add_argument("--em-iters", type=int, default=50, help="EM iterations for Kalman ISS.")
    parser.add_argument("--em-tol", type=float, default=1e-4, help="EM convergence tolerance.")
    parser.add_argument("--em-ridge", type=float, default=1e-6, help="Ridge stabiliser for ISS EM.")
    parser.add_argument("--macro-eps", type=float, default=0.25, help="Tolerance for continuous macrostate reconstruction.")
    parser.add_argument("--macro-bins", type=int, default=3, help="Quantile bins per macro dimension for transition symbols.")
    parser.add_argument(
        "--macro-symboliser",
        type=str,
        choices=["quantile"],
        default="quantile",
        help="Discretiser used to turn macro projections into transition symbols.",
    )
    parser.add_argument(
        "--macro-projection",
        type=str,
        choices=["pca", "random", "psi_opt"],
        default="pca",
        help="Projection family used for V_t = L X_t in continuous runs.",
    )
    parser.add_argument(
        "--iss-mode",
        type=str,
        choices=["steady_state", "time_varying"],
        default="steady_state",
        help="Filtering mode used for ISS predictions.",
    )
    parser.add_argument(
        "--allow-time-varying-fallback",
        action="store_true",
        help="If steady-state DARE fails, fall back to time-varying Kalman gains.",
    )
    parser.add_argument("--steady-state-tol", type=float, default=1e-9, help="Steady-state Riccati convergence tolerance.")
    parser.add_argument("--steady-state-max-iter", type=int, default=10_000, help="Max iterations for steady-state Riccati solver.")
    parser.add_argument("--steady-state-ridge", type=float, default=1e-9, help="Ridge stabiliser for steady-state Riccati solver.")
    parser.add_argument("--compute-psi", action="store_true", help="Optimise ISS Psi over coarse-graining L.")
    parser.add_argument(
        "--psi-optimiser",
        dest="psi_optimiser",
        type=str,
        choices=["random", "torch_adam"],
        default="random",
        help="Optimiser used for Psi coarse-graining.",
    )
    parser.add_argument(
        "--psi-optimizer",
        dest="psi_optimiser",
        type=str,
        choices=["random", "torch_adam"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--psi-restarts", type=int, default=12, help="Number of restarts for Psi optimisation.")
    parser.add_argument("--psi-iters", type=int, default=120, help="Iterations per restart for Psi optimisation.")
    parser.add_argument("--psi-lr", type=float, default=0.03, help="Learning rate for torch Adam Psi optimisation.")
    parser.add_argument("--psi-step-scale", type=float, default=0.2, help="Step scale for random Psi search.")
    parser.add_argument("--psi-tol", type=float, default=1e-8, help="Tolerance used in Psi solvers.")
    parser.add_argument("--psi-max-iter", type=int, default=4000, help="Max Riccati/Lyapunov iterations for Psi.")
    parser.add_argument("--psi-ridge", type=float, default=1e-8, help="Covariance ridge for Psi computations.")

    parser.add_argument("--data-path", type=Path, default=None, help="Path to .npy/.csv/.txt time series.")
    parser.add_argument("--data-column", type=int, default=None, help="Column index for multi-column files.")
    parser.add_argument("--data-columns", nargs="+", type=int, default=None, help="Column indices for multi-column files.")

    parser.add_argument("--lgssm-a", type=float, default=0.92, help="AR coefficient for linear_gaussian_ssm.")
    parser.add_argument("--lgssm-c", type=float, default=1.0, help="Observation loading for linear_gaussian_ssm.")
    parser.add_argument("--lgssm-process-std", type=float, default=0.35, help="Process noise std.")
    parser.add_argument("--lgssm-obs-std", type=float, default=0.25, help="Observation noise std.")
    parser.add_argument("--lgssm-init-std", type=float, default=1.0, help="Initial latent std.")
    parser.add_argument("--lgssm-latent-dim", type=int, default=2, help="Latent state dimension for linear_gaussian_ssm.")
    parser.add_argument("--lgssm-obs-dim", type=int, default=3, help="Observation dimension p for linear_gaussian_ssm.")
    parser.add_argument("--lgssm-coupling-std", type=float, default=0.08, help="Off-diagonal coupling scale in latent dynamics.")

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

    parser.add_argument("--save-transitions", action="store_true", help="Save reconstructed macrostate transitions.")
    parser.add_argument("--show-transitions-for", type=str, default=None, help="Representation name to print.")
    parser.add_argument("--show-seed", type=int, default=None, help="Seed for transition inspection.")
    parser.add_argument("--show-flip-p", type=float, default=None, help="flip_p for transition inspection.")
    parser.add_argument("--show-subsample-step", type=int, default=None, help="subsample step for transition inspection.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="CLI logging verbosity.",
    )

    return parser.parse_args()


def _build_process(args: argparse.Namespace):
    if args.process == "continuous_file":
        if args.data_path is None:
            raise ValueError("--data-path is required for --process continuous_file")
        return PROCESS_REGISTRY[args.process](
            path=args.data_path,
            column=args.data_column,
            columns=tuple(args.data_columns) if args.data_columns is not None else None,
        )
    if args.process == "linear_gaussian_ssm":
        return PROCESS_REGISTRY[args.process](
            a=args.lgssm_a,
            c=args.lgssm_c,
            process_std=args.lgssm_process_std,
            obs_std=args.lgssm_obs_std,
            init_std=args.lgssm_init_std,
            latent_dim=args.lgssm_latent_dim,
            obs_dim=args.lgssm_obs_dim,
            coupling_std=args.lgssm_coupling_std,
        )
    return PROCESS_REGISTRY[args.process]()


def _build_representations(args: argparse.Namespace) -> list[Representation]:
    if args.reconstructor == "kalman_iss":
        return [ISSDim(d=d, dv=dv) for d, dv in product(args.ks, args.dvs)]
    if args.noisy:
        noise = bernoulli_noise(length=args.length, seed=args.noise_seed)
        return [LastKWithNoise(k=k, noise=noise) for k in args.ks]
    return [LastK(k=k) for k in args.ks]


def main() -> None:
    args = _parse_args()
    configure_logging(getattr(logging, args.log_level))

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
    if args.compute_psi and args.reconstructor != "kalman_iss":
        raise ValueError("--compute-psi is only supported with --reconstructor kalman_iss.")
    if args.psi_restarts < 1:
        raise ValueError("--psi-restarts must be >= 1.")
    if args.psi_iters < 1:
        raise ValueError("--psi-iters must be >= 1.")
    if any(k < 1 for k in args.ks):
        raise ValueError("All --ks values must be >= 1.")
    if any(dv < 1 for dv in args.dvs):
        raise ValueError("All --dvs values must be >= 1.")
    if args.data_column is not None and args.data_columns is not None:
        raise ValueError("Use either --data-column or --data-columns, not both.")
    if args.macro_eps < 0.0:
        raise ValueError("--macro-eps must be >= 0.")
    if args.macro_bins < 1:
        raise ValueError("--macro-bins must be >= 1.")
    if args.steady_state_tol <= 0.0:
        raise ValueError("--steady-state-tol must be > 0.")
    if args.steady_state_max_iter < 1:
        raise ValueError("--steady-state-max-iter must be >= 1.")
    if args.steady_state_ridge < 0.0:
        raise ValueError("--steady-state-ridge must be >= 0.")

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
            macro_eps=args.macro_eps,
            macro_bins=args.macro_bins,
            macro_symboliser=args.macro_symboliser,
            projection_mode=args.macro_projection,
            iss_mode=args.iss_mode,
            allow_time_varying_fallback=args.allow_time_varying_fallback,
            steady_state_tol=args.steady_state_tol,
            steady_state_max_iter=args.steady_state_max_iter,
            steady_state_ridge=args.steady_state_ridge,
            compute_psi=args.compute_psi,
            psi_optimiser=args.psi_optimiser,
            psi_restarts=args.psi_restarts,
            psi_iterations=args.psi_iters,
            psi_lr=args.psi_lr,
            psi_step_scale=args.psi_step_scale,
            psi_tol=args.psi_tol,
            psi_max_iter=args.psi_max_iter,
            psi_ridge=args.psi_ridge,
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
        "dvs": args.dvs,
        "flip_ps": flip_ps,
        "subsample_steps": subsample_steps,
        "noisy_representation": args.noisy,
        "noise_seed": args.noise_seed if args.noisy else None,
        "save_transitions": args.save_transitions,
        "transitions_rep_name": args.show_transitions_for if args.save_transitions else None,
        "condition_id_style": args.condition_id_style,
        "data_path": str(args.data_path) if args.data_path is not None else None,
        "data_column": args.data_column,
        "data_columns": args.data_columns,
        "linear_gaussian_ssm": {
            "a": args.lgssm_a,
            "c": args.lgssm_c,
            "process_std": args.lgssm_process_std,
            "obs_std": args.lgssm_obs_std,
            "init_std": args.lgssm_init_std,
            "latent_dim": args.lgssm_latent_dim,
            "obs_dim": args.lgssm_obs_dim,
            "coupling_std": args.lgssm_coupling_std,
        }
        if args.process == "linear_gaussian_ssm"
        else None,
        "kalman_iss": {
            "em_iters": args.em_iters,
            "em_tol": args.em_tol,
            "em_ridge": args.em_ridge,
            "macro_eps": args.macro_eps,
            "macro_bins": args.macro_bins,
            "macro_symboliser": args.macro_symboliser,
            "macro_projection": args.macro_projection,
            "iss_mode": args.iss_mode,
            "allow_time_varying_fallback": args.allow_time_varying_fallback,
            "steady_state_tol": args.steady_state_tol,
            "steady_state_max_iter": args.steady_state_max_iter,
            "steady_state_ridge": args.steady_state_ridge,
            "compute_psi": args.compute_psi,
            "psi_optimiser": args.psi_optimiser,
            "psi_restarts": args.psi_restarts,
            "psi_iterations": args.psi_iters,
            "psi_lr": args.psi_lr,
            "psi_step_scale": args.psi_step_scale,
            "psi_tol": args.psi_tol,
            "psi_max_iter": args.psi_max_iter,
            "psi_ridge": args.psi_ridge,
        }
        if args.reconstructor == "kalman_iss"
        else None,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    save_json(args.outdir / "config.json", config)
    LOGGER.info(
        "Running PRISM | process=%s reconstructor=%s reps=%d seeds=%d outdir=%s",
        args.process,
        args.reconstructor,
        len(representations),
        len(args.seeds),
        args.outdir,
    )
    total_conditions = len(flip_ps) * len(subsample_steps)
    condition_index = 0

    for flip_p in flip_ps:
        for step in subsample_steps:
            condition_index += 1
            LOGGER.info(
                "Condition %d/%d | flip_p=%g subsample_step=%d",
                condition_index,
                total_conditions,
                flip_p,
                step,
            )
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
            LOGGER.info("Condition %d/%d complete", condition_index, total_conditions)

    LOGGER.info("Run complete | metrics at %s", args.outdir / "runs.csv")

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
