from pathlib import Path
from typing import List, Optional

from prism.metrics import log_loss, statistical_complexity, n_states
from prism.utils.io import save_csv, save_json
from prism.metrics.graph import to_edge_list, to_dot, save_dot
from prism.processes.protocols import Process
from prism.reconstruction.protocols import Reconstructor
from prism.representations.protocols import Representation


def run_experiment(
    *,
    process: Process,
    reconstructor: Reconstructor,
    representations: List[Representation],
    length: int,
    train_frac: float,
    seeds: List[int],
    outdir: Path,
    save_transitions: bool = False,
    transitions_rep_name: Optional[str] = None,
) -> None:
    """Run an experiement and write reproducible artefacts

    Args:
        process: Stochastic process to sample from
        reconstructor: Model reconstructor for fitting
        representations (List): List of representations to evaluate
        length (int): Length of sequence to sample
        train_frac (float): Fraction of data to use for training
        seed (int): Random seed for reproducibility
        outdir (Path): Output directory for results
        save_transitions (bool, optional): Whether to save state transitions, defaults to False
        transitions_rep_name (Optional[str], optional): Representation name for transitions, defaults to None.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Keep a single metrics.csv and append per seed
    metrics_path = outdir / "metrics.csv"

    # Fixed column order
    fieldnames = ["seed", "process", "representation", "reconstructor", "logloss", "n_states", "C_mu"]
    
    for seed in seeds:
        sample = process.sample(length=length, seed=seed)
        x = sample.x

        split = int(len(x) * train_frac)
        x_train, x_test = x[:split], x[split:]

        rows = []
        for rep in representations:
            model = reconstructor.fit(x_train, rep, seed=seed)

            rows.append(
                {
                    "seed": seed,
                    "process": process.name,
                    "representation": rep.name,
                    "reconstructor": reconstructor.name,
                    "logloss": log_loss(x_test, rep, model),
                    "n_states": n_states(model),
                    "C_mu": statistical_complexity(model),
                }
            )

            if save_transitions:
                if transitions_rep_name is None or rep.name == transitions_rep_name:
                    edges = to_edge_list(model)
                    # Save JSON edge list
                    save_json(outdir / f"transitions_{rep.name}_seed{seed}.json", edges)
                    # Save DOT for Graphviz
                    dot = to_dot(
                        edges,
                        graph_name=f"{process.name}_{rep.name}_seed{seed}",
                        label=f"{process.name} | {rep.name} | seed={seed}",
                    )
                    save_dot(outdir / f"transitions_{rep.name}_seed{seed}.dot", dot)

        # Append per seed
        save_csv(metrics_path, rows, append=True, fieldnames=fieldnames)

    # One config for the whole sweep
    save_json(
        outdir / "config.json",
        {
            "process": process.name,
            "reconstructor": reconstructor.name,
            "length": length,
            "train_frac": train_frac,
            "seeds": seeds,
            "representations": [rep.name for rep in representations],
            "save_transitions": save_transitions,
            "transitions_rep_name": transitions_rep_name,
        },
    )