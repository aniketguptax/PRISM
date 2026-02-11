import argparse
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from prism.analysis.continuous_common import (
    available_projection_modes,
    build_axes,
    filter_rows,
    finite_min_max,
    metric_column,
    mode_matrix,
    read_rows,
)


DEFAULT_METRICS = ["gaussian_logloss", "n_states", "C_mu_empirical", "psi_opt"]


def _metric_label(metric_col: str) -> str:
    label_map = {
        "gaussian_logloss_mean": "Gaussian log-loss (nats)",
        "n_states_mean": "Macrostate count |M|",
        "C_mu_empirical_mean": "$C_\\mu$ (empirical)",
        "psi_opt_mean": "Optimised ISS $\\Psi$",
    }
    return label_map.get(metric_col, metric_col)


def _plot_heatmap(
    *,
    matrix: np.ndarray,
    ks: Sequence[int],
    dvs: Sequence[int],
    title: str,
    color_label: str,
    outpath: Path,
    vmin: float | None,
    vmax: float | None,
) -> None:
    fig_w = max(4.8, 2.4 + 0.55 * len(ks))
    fig_h = max(3.4, 2.2 + 0.45 * len(dvs))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("ISS state dimension $d$")
    ax.set_ylabel("Macro dimension $d_V$")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks(range(len(dvs)))
    ax.set_yticklabels([str(dv) for dv in dvs])
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(color_label)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


def _normalise_formats(values: Sequence[str]) -> List[str]:
    valid = {"png", "pdf"}
    formats: List[str] = []
    for raw in values:
        fmt = raw.strip().lower()
        if fmt not in valid:
            raise ValueError(f"Unsupported format {raw!r}. Expected one of {sorted(valid)}")
        if fmt not in formats:
            formats.append(fmt)
    return formats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Run root containing summary_by_condition.csv")
    parser.add_argument("--infile", type=str, default="summary_by_condition.csv")
    parser.add_argument("--base-process", type=str, default=None)
    parser.add_argument("--condition-id", type=str, default=None)
    parser.add_argument("--subsample-step", type=int, default=1)
    parser.add_argument("--flip-p", type=float, default=0.0)
    parser.add_argument("--projection-modes", nargs="+", default=None)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--shared-scale", action="store_true", help="Use shared colour range across projection modes.")
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    summary = args.root / args.infile
    if not summary.exists():
        raise FileNotFoundError(f"Missing {summary}")

    formats = _normalise_formats(args.formats)
    rows = filter_rows(
        read_rows(summary),
        base_process=args.base_process,
        condition_id=args.condition_id,
        subsample_step=args.subsample_step,
        flip_p=args.flip_p,
        projection_modes=args.projection_modes,
    )
    if not rows:
        raise ValueError("No rows match the requested filters for continuous heatmaps.")

    modes = args.projection_modes if args.projection_modes else available_projection_modes(rows)
    ks, dvs = build_axes(rows)
    if not ks or not dvs:
        raise ValueError("Cannot build heatmaps: missing valid k/dv values in filtered rows.")

    outdir = args.outdir or (args.root / "figures" / "continuous_heatmaps")
    for metric in args.metrics:
        metric_col = metric_column(rows, metric)
        mode_to_matrix = {
            mode: mode_matrix(rows, projection_mode=mode, metric_col=metric_col, ks=ks, dvs=dvs)
            for mode in modes
        }
        vmin = vmax = None
        if args.shared_scale:
            vmin, vmax = finite_min_max(mode_to_matrix.values())
        for mode, matrix in mode_to_matrix.items():
            if not np.isfinite(matrix).any():
                print(f"Skipping {mode}/{metric_col}: no finite values.")
                continue
            for fmt in formats:
                outpath = outdir / f"{mode}_{metric_col}_heatmap.{fmt}"
                _plot_heatmap(
                    matrix=matrix,
                    ks=ks,
                    dvs=dvs,
                    title=f"{mode} | {metric_col}",
                    color_label=_metric_label(metric_col),
                    outpath=outpath,
                    vmin=vmin,
                    vmax=vmax,
                )


if __name__ == "__main__":
    main()
