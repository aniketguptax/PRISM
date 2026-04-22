"""Plot recovery diagnostics for the block-modular sweep."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _hierarchy_plot(rows: list[dict[str, str]], obs_design: str, outpath: Path) -> None:
    iss_rows = [
        row
        for row in rows
        if row.get("obs_design") == obs_design and row.get("method", "").startswith("iss_")
    ]
    if not iss_rows:
        return

    builders = sorted({row["method"].replace("iss_", "") for row in iss_rows})
    couplings = sorted({float(row["coupling"]) for row in iss_rows})
    eps_macros = sorted({float(row["method_param"]) for row in iss_rows})

    available = set(iss_rows[0].keys())
    metrics: list[tuple[str, str]] = [("n_states", "State count |M|")]
    for key, label in (
        ("ari_joint", "ARI vs joint regime"),
        ("ari_slow_block", "ARI vs slow block"),
        ("ari_phase_block", "ARI vs rotational phase"),
    ):
        if key in available:
            metrics.append((key, label))
    if "ari_ground_truth" in available and "ari_joint" not in available:
        metrics.append(("ari_ground_truth", "ARI vs best ground truth"))
    metrics.extend(
        [
            ("unifilarity", "Unifilarity"),
            ("branch_entropy", "Branch entropy (bits)"),
            ("macro_build_time_s", "Build time (s)"),
        ]
    )

    fig, axes = plt.subplots(
        len(metrics),
        len(builders),
        figsize=(3.0 * len(builders), 2.2 * len(metrics)),
        squeeze=False,
    )
    cmap = plt.get_cmap("viridis")
    for col, builder in enumerate(builders):
        builder_rows = [row for row in iss_rows if row["method"] == f"iss_{builder}"]
        for row_idx, (metric_key, title) in enumerate(metrics):
            ax = axes[row_idx][col]
            for coupling_idx, coupling in enumerate(couplings):
                ys: list[float] = []
                for eps in eps_macros:
                    matched = [
                        row
                        for row in builder_rows
                        if abs(float(row["coupling"]) - coupling) < 1e-9
                        and abs(float(row["method_param"]) - eps) < 1e-9
                    ]
                    values = [_f(row.get(metric_key, "")) for row in matched]
                    clean = [value for value in values if value is not None]
                    ys.append(float(np.mean(clean)) if clean else np.nan)
                ax.plot(
                    eps_macros,
                    ys,
                    marker="o",
                    color=cmap(coupling_idx / max(len(couplings) - 1, 1)),
                    label=f"ε={coupling:g}",
                )
            if row_idx == 0:
                ax.set_title(builder, fontsize=9)
            if col == 0:
                ax.set_ylabel(title, fontsize=8)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("eps_macro", fontsize=8)
            ax.tick_params(labelsize=7)
            if metric_key == "n_states":
                ax.set_yscale("log")

    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(couplings), 6),
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(f"Block-modular recovery hierarchy ({obs_design} mixing)", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _baseline_plot(
    rows: list[dict[str, str]],
    obs_design: str,
    outpath: Path,
    *,
    target_key: str = "ari_joint",
    target_label: str = "ARI vs joint regime",
) -> None:
    obs_rows = [row for row in rows if row.get("obs_design") == obs_design]
    if not obs_rows:
        return
    if target_key not in obs_rows[0]:
        target_key = "ari_ground_truth"
        target_label = "ARI vs best ground truth"

    couplings = sorted({float(row["coupling"]) for row in obs_rows})

    iss_best: dict[float, float] = {}
    for coupling in couplings:
        iss_for_coupling = [
            row
            for row in obs_rows
            if abs(float(row["coupling"]) - coupling) < 1e-9
            and row.get("method", "").startswith("iss_")
        ]
        agg: dict[tuple[str, str], list[float]] = {}
        for row in iss_for_coupling:
            ari = _f(row.get(target_key, ""))
            if ari is None:
                continue
            agg.setdefault((row["method"], row["method_param"]), []).append(ari)
        if not agg:
            continue
        _, best_values = max(agg.items(), key=lambda item: float(np.mean(item[1])))
        iss_best[coupling] = float(np.mean(best_values))

    pca_rows = [row for row in obs_rows if row.get("method") == "pca_kmeans"]
    pca_by_k: dict[int, dict[float, float]] = {}
    for k in sorted({int(float(row["method_param"])) for row in pca_rows}):
        per_coupling: dict[float, list[float]] = {}
        for row in pca_rows:
            if int(float(row["method_param"])) != k:
                continue
            ari = _f(row.get(target_key, ""))
            if ari is None:
                continue
            coupling = float(row["coupling"])
            per_coupling.setdefault(coupling, []).append(ari)
        pca_by_k[k] = {
            coupling: float(np.mean(values))
            for coupling, values in per_coupling.items()
            if values
        }

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(
        list(iss_best.keys()),
        list(iss_best.values()),
        marker="o",
        color="black",
        label="ISS (best builder/eps)",
    )
    cmap = plt.get_cmap("plasma")
    for idx, (k, mapping) in enumerate(sorted(pca_by_k.items())):
        xs = sorted(mapping.keys())
        ax.plot(
            xs,
            [mapping[coupling] for coupling in xs],
            marker="s",
            color=cmap(idx / max(len(pca_by_k) - 1, 1)),
            label=f"PCA+k-means k={k}",
        )

    ax.set_xlabel("Coupling ε")
    ax.set_ylabel(target_label)
    ax.set_title(f"ISS vs PCA+k-means ({obs_design})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Directory containing recovery.csv")
    args = parser.parse_args()

    csv_path = args.root / "recovery.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows = _read(csv_path)
    obs_designs = sorted({row.get("obs_design", "") for row in rows if row.get("obs_design")})
    figures_dir = args.root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("ari_joint", "ARI vs joint regime", "joint"),
        ("ari_slow_block", "ARI vs slow block", "slow"),
        ("ari_phase_block", "ARI vs rotational phase", "phase"),
    ]
    for design in obs_designs:
        _hierarchy_plot(rows, design, figures_dir / f"hierarchy_{design}.png")
        for target_key, target_label, suffix in targets:
            _baseline_plot(
                rows,
                design,
                figures_dir / f"baseline_{design}_{suffix}.png",
                target_key=target_key,
                target_label=target_label,
            )
        _baseline_plot(rows, design, figures_dir / f"baseline_{design}.png")

    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
