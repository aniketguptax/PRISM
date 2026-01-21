import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_vs_k(
    summary_csv: Path,
    metric: str,
    outpath: Path,
    subsample_step: int = 1
) -> None:
    df = pd.read_csv(summary_csv)

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df.columns:
        raise ValueError(
            f"{mean_col} not found in {summary_csv}. "
            f"Available cols: {list(df.columns)[:25]}..."
        )

    # Filter for a single subsample step for a clean plot
    if "subsample_step" not in df.columns:
        raise ValueError("summary CSV missing 'subsample_step' column.")
    df = df[df["subsample_step"] == subsample_step].copy()

    if df.empty:
        raise ValueError(f"No rows for subsample_step={subsample_step} in {summary_csv}")
    
    # Fresh figure per metric
    plt.figure(figsize=(4.8, 3.2))

    # One line per flip_p
    n_lines = 0
    for flip_p, g in df.groupby("flip_p"):
        g = g.sort_values("k")
        x = g["k"].to_numpy()
        y = g[mean_col].to_numpy()
        label = f"flip_p={flip_p:g}"

        if std_col in g.columns:
            yerr = g[std_col].to_numpy()
            plt.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o-",
                capsize=3,
                label=label,
            )
        else:
            plt.errorbar(
                x,
                y,
                fmt="o-",
                capsize=3,
                label=label,
            )
        n_lines += 1

    plt.xlabel("Representation length $k$")
    ylabel = {
        "branch_entropy": "Branching entropy (bits)",
        "unifilarity_score": "Unifilarity score",
        "logloss": "Log-loss (nats)",
    }.get(metric, metric)
    plt.ylabel(ylabel)

    if n_lines > 1:
        plt.legend(fontsize=8)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()
    print(f"Wrote {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Grid root folder containing summary_by_condition.csv",
    )
    parser.add_argument("--subsample-step", type=int, default=1)
    args = parser.parse_args()

    summary_csv = args.root / "summary_by_condition.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing {summary_csv}. Run summarise_grid.py first.")

    figs_dir = args.root / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Branch entropy plot
    plot_metric_vs_k(
        summary_csv,
        metric="branch_entropy",
        outpath=figs_dir / f"branch_entropy_vs_k_sub{args.subsample_step}.pdf",
        subsample_step=args.subsample_step,
    )

    # Unifilarity plot
    plot_metric_vs_k(
        summary_csv,
        metric="unifilarity_score",
        outpath=figs_dir / f"unifilarity_vs_k_sub{args.subsample_step}.pdf",
        subsample_step=args.subsample_step,
    )


if __name__ == "__main__":
    main()