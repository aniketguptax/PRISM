from pathlib import Path
from typing import List, Optional, Dict, Any

from prism.metrics import log_loss, statistical_complexity, n_states, unifilarity_score
from prism.metrics.branching import mean_branching_entropy_weighted
from prism.utils.io import save_csv, save_json
from prism.metrics.graph import DotStyle, dot_to_png, to_edge_list, to_dot, save_dot
from prism.processes.protocols import Process
from prism.reconstruction.protocols import Reconstructor
from prism.representations.protocols import Representation

def _extract_k(rep_name: str) -> Optional[int]:
    # e.g. "last_3" -> 3, "last_5_noisy" -> 5
    import re
    m = re.search(r"last_(\d+)", rep_name)
    return int(m.group(1)) if m else None

def _cond_fields(condition: Optional[Dict[str, Any]], process_name_fallback: str) -> Dict[str, Any]:
    cond = condition or {}
    base_process = str(cond.get("base_process", process_name_fallback))
    flip_p = float(cond.get("flip_p", 0.0))
    subsample_step = int(cond.get("subsample_step", 1))
    wrappers = str(cond.get("wrappers", ""))
    condition_id = str(cond.get("condition_id", f"flip{flip_p:g}_sub{subsample_step}"))
    return {
        "base_process": base_process,
        "flip_p": flip_p,
        "subsample_step": subsample_step,
        "wrappers": wrappers,
        "condition_id": condition_id
    }

def run_experiment(
    process: Process,
    reconstructor: Reconstructor,
    representations: List[Representation],
    length: int,
    train_frac: float,
    seeds: List[int],
    outdir: Path,
    condition: Optional[Dict[str, Any]] = None,
    save_transitions: bool = False,
    transitions_rep_name: Optional[str] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = outdir / "runs.csv"
    fieldnames = [
        "seed",
        "length",
        "train_frac",
        "process",
        "base_process",
        "eps",
        "flip_p",
        "subsample_step",
        "wrappers",
        "condition_id",
        "representation",
        "k",
        "reconstructor",
        "logloss",
        "n_states",
        "C_mu_empirical",
        "unifilarity_score",
        "branch_entropy"
    ]
    
    cond = _cond_fields(condition, process.name)
    condition_id = cond["condition_id"]
    
    # transitions dir
    tdir = outdir / "transitions"
    
    for seed in seeds:
        sample = process.sample(length=length, seed=seed)
        x = sample.x

        split = int(len(x) * train_frac)
        x_train, x_test = x[:split], x[split:]

        rows = []
        for rep in representations:
            model = reconstructor.fit(x_train, rep, seed=seed)
            k = _extract_k(rep.name)
            eps = getattr(reconstructor, "eps", None)
                        
            rows.append({
                "seed": seed,
                "length": length,
                "train_frac": train_frac,
                "process": process.name,
                "eps": eps,
                **cond,
                "representation": rep.name,
                "k": k,
                "reconstructor": reconstructor.name,
                "logloss": log_loss(x_test, rep, model),
                "n_states": n_states(model),
                "C_mu_empirical": statistical_complexity(model),
                "unifilarity_score": unifilarity_score(model),
                "branch_entropy": mean_branching_entropy_weighted(model, log_base=2.0),
            })

            if save_transitions and (transitions_rep_name is None or rep.name == transitions_rep_name):
                    edges = to_edge_list(model)
                    tdir.mkdir(exist_ok=True)
                    
                    json_path = tdir / f"{condition_id}__transitions_{rep.name}_seed{seed}.json"
                    dot_path = tdir / f"{condition_id}__transitions_{rep.name}_seed{seed}.dot"
                    png_path = tdir / f"{condition_id}__transitions_{rep.name}_seed{seed}.png"
                    
                    save_json(json_path, edges)
                    dot = to_dot(
                        edges,
                        graph_name=f"{process.name}_{rep.name}_seed{seed}",
                        label=f"{process.name} | {rep.name} | seed={seed} | {condition_id}",
                        style=DotStyle(rankdir="LR"),
                    )


                    save_dot(dot_path, dot)
                    dot_to_png(dot_path, png_path)

        save_csv(metrics_path, rows, append=True, fieldnames=fieldnames)