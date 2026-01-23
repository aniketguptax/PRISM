import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="Run root directory (contains runs.csv)")
    p.add_argument("--subsample-step", type=int, default=1, help="Which subsample step for k-plots")
    p.add_argument("--metrics", nargs="+", default=["branch_entropy", "unifilarity_score", "logloss"],
                   help="Metrics to plot vs k")
    p.add_argument("--phase", action="store_true", help="Also produce phase diagrams")
    
    # filters
    p.add_argument("--base-process", type=str, default=None, help="Filter for plots")
    p.add_argument("--representation", type=str, default=None, help="Filter for plots")
    p.add_argument("--condition-id", type=str, default=None, help="Filter for plots")
    p.add_argument("--flip-ps", nargs="+", type=float, default=None, help="Only plot these flip probabilities")
    
    args = p.parse_args()
    root = args.root
    
    if not root.exists():
        raise FileNotFoundError(root)

    # summarise
    run([sys.executable, "-m", "prism.analysis.summarise", "--root", str(root)])

    # plot_k for requested metrics
    plot_k_cmd = [sys.executable, "-m", "prism.analysis.plot_k", "--root", str(root), "--subsample-step", str(args.subsample_step),
                  "--metrics", *args.metrics]
    if args.base_process is not None:
        plot_k_cmd += ["--base-process", args.base_process]
    if args.representation is not None:
        plot_k_cmd += ["--representation", args.representation]
    if args.condition_id is not None:
        plot_k_cmd += ["--condition-id", args.condition_id]
    if args.flip_ps is not None:
        plot_k_cmd += ["--flip-ps", *(str(p) for p in args.flip_ps)]
    run(plot_k_cmd)

    # phase diagram
    if args.phase:
        phase_cmd = [sys.executable, "-m", "prism.analysis.phase_diagram", "--root", str(root)]
        if args.base_process is not None:
            phase_cmd += ["--base-process", args.base_process]
        if args.representation is not None:
            phase_cmd += ["--representation", args.representation]
        if args.condition_id is not None:
            phase_cmd += ["--condition-id", args.condition_id]
        run(phase_cmd)

    print("All figures done.")


if __name__ == "__main__":
    main()