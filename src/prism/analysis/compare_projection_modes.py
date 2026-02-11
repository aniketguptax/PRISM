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


DEFAULT_MODES = ["pca", "random", "psi_opt"]
DEFAULT_METRICS = ["gaussian_logloss", "n_states", "C_mu_empirical", "psi_opt"]


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


def _plot_mode_comparison(
    *,
    modes: Sequence[str],
    matrices: dict[str, np.ndarray],
    ks: Sequence[int],
    dvs: Sequence[int],
    metric_col: str,
    outpath: Path,
    vmin: float,
    vmax: float,
) -> None:
    fig_w = max(4.6 * len(modes), 9.0)
    fig_h = max(3.6, 2.3 + 0.45 * len(dvs))
    fig, axes = plt.subplots(1, len(modes), figsize=(fig_w, fig_h), sharex=True, sharey=True)
    if len(modes) == 1:
        axes = [axes]

    color_image = None
    for axis, mode in zip(axes, modes):
        matrix = matrices.get(mode)
        if matrix is None or not np.isfinite(matrix).any():
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(mode)
            axis.set_xticks([])
            axis.set_yticks([])
            continue
        masked = np.ma.masked_invalid(matrix)
        color_image = axis.imshow(
            masked,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(mode)
        axis.set_xticks(range(len(ks)))
        axis.set_xticklabels([str(k) for k in ks])
        axis.set_yticks(range(len(dvs)))
        axis.set_yticklabels([str(dv) for dv in dvs])
        axis.set_xlabel("ISS state dimension $d$")

    axes[0].set_ylabel("Macro dimension $d_V$")
    if color_image is not None:
        cbar = fig.colorbar(color_image, ax=axes, shrink=0.9)
        cbar.set_label(metric_col)
    fig.suptitle(f"Projection-mode comparison: {metric_col}")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Run root containing summary_by_condition.csv")
    parser.add_argument("--infile", type=str, default="summary_by_condition.csv")
    parser.add_argument("--base-process", type=str, default=None)
    parser.add_argument("--condition-id", type=str, default=None)
    parser.add_argument("--subsample-step", type=int, default=1)
    parser.add_argument("--flip-p", type=float, default=0.0)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
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
        projection_modes=None,
    )
    if not rows:
        raise ValueError("No rows match the requested filters for projection-mode comparison.")

    available_modes = set(available_projection_modes(rows))
    modes = [mode for mode in args.modes if mode in available_modes]
    if not modes:
        raise ValueError(
            f"None of requested modes {args.modes} are present in filtered rows. Available: {sorted(available_modes)}"
        )

    ks, dvs = build_axes(rows)
    if not ks or not dvs:
        raise ValueError("Cannot build comparison plots: missing valid k/dv values in filtered rows.")

    outdir = args.outdir or (args.root / "figures" / "projection_mode_comparison")
    for metric in args.metrics:
        metric_col = metric_column(rows, metric)
        matrices = {mode: mode_matrix(rows, projection_mode=mode, metric_col=metric_col, ks=ks, dvs=dvs) for mode in modes}
        vmin, vmax = finite_min_max(matrices.values())
        if vmin is None or vmax is None:
            print(f"Skipping {metric_col}: no finite values across selected modes.")
            continue
        for fmt in formats:
            outpath = outdir / f"compare_projection_modes_{metric_col}.{fmt}"
            _plot_mode_comparison(
                modes=modes,
                matrices=matrices,
                ks=ks,
                dvs=dvs,
                metric_col=metric_col,
                outpath=outpath,
                vmin=vmin,
                vmax=vmax,
            )


if __name__ == "__main__":
    main()
