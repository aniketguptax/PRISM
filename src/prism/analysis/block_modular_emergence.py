"""Summarise CE 2.0 on block-modular recovery runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np

from prism.analysis.causal_emergence import (
    PathRung,
    causal_primitives,
    cp_path_from_label_chain,
    delta_cp,
    emergent_complexity,
    macro_tpm_from_labels,
    total_ce,
)

GROUND_TRUTH_NPZ_KEYS = {
    "joint": "gt_joint_full",
    "slow": "gt_slow_block_full",
    "phase": "gt_phase_block_full",
    "block_attribution": "ground_truth_full",
}


def _read_recovery_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _list_npz(root: Path) -> list[Path]:
    return sorted(root.glob("labels_eps*.npz"))


def _parse_npz_key(path: Path) -> tuple[float, int, str]:
    stem = path.stem
    parts = stem.split("_")
    coupling = float(parts[1][3:])
    seed = int(parts[2][4:])
    design = "_".join(parts[3:])
    return coupling, seed, design


def _align_labels(labels: np.ndarray, target_len: int) -> np.ndarray:
    if labels.shape[0] == target_len:
        return labels
    if labels.shape[0] > target_len:
        return labels[:target_len]
    pad = np.full(target_len - labels.shape[0], labels[-1], dtype=labels.dtype)
    return np.concatenate([labels, pad], axis=0)


def _dense_relabel(labels: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(int)


def _path_for_builder(npz: np.lib.npyio.NpzFile, builder: str) -> list[np.ndarray]:
    prefix = f"iss_{builder}_eps"
    entries: list[tuple[float, np.ndarray]] = []
    for key in npz.files:
        if not key.startswith(prefix):
            continue
        try:
            eps_value = float(key[len(prefix) :])
        except ValueError:
            continue
        entries.append((eps_value, _dense_relabel(np.asarray(npz[key], dtype=int))))
    if not entries:
        return []
    entries.sort(key=lambda item: item[0])
    entries.sort(key=lambda item: -(int(item[1].max()) + 1))

    chain: list[np.ndarray] = []
    seen_sizes: set[int] = set()
    for _, labels in entries:
        n_states = int(labels.max()) + 1
        if n_states in seen_sizes:
            continue
        seen_sizes.add(n_states)
        chain.append(labels)
    return chain


def _path_for_pca(npz: np.lib.npyio.NpzFile) -> list[np.ndarray]:
    entries: list[tuple[int, np.ndarray]] = []
    for key in npz.files:
        if not key.startswith("pca_kmeans_k"):
            continue
        try:
            n_states = int(key[len("pca_kmeans_k") :])
        except ValueError:
            continue
        entries.append((n_states, _dense_relabel(np.asarray(npz[key], dtype=int))))
    entries.sort(key=lambda item: -item[0])
    return [labels for _, labels in entries]


def _ground_truth_path(
    npz: np.lib.npyio.NpzFile,
    target_len: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, key in GROUND_TRUTH_NPZ_KEYS.items():
        if key not in npz.files:
            continue
        labels = _dense_relabel(np.asarray(npz[key], dtype=int))
        out[name] = _align_labels(labels, target_len)
    return out


def _summarise_path(
    chain: list[np.ndarray],
    *,
    use_observed_distribution: bool,
) -> dict[str, object]:
    if not chain:
        return {
            "n_rungs": 0,
            "rungs": [],
            "delta_cp": [],
            "total_ce": 0.0,
            "emergent_complexity": 0.0,
            "emergent_complexity_raw_bits": 0.0,
        }
    rungs = cp_path_from_label_chain(
        chain,
        use_observed_distribution=use_observed_distribution,
    )
    deltas = delta_cp(rungs)
    return {
        "n_rungs": len(rungs),
        "rungs": rungs,
        "delta_cp": deltas,
        "total_ce": total_ce(rungs),
        "emergent_complexity": emergent_complexity(deltas, normalise=True),
        "emergent_complexity_raw_bits": emergent_complexity(deltas, normalise=False),
    }


def _single_rung_cp(labels: np.ndarray, *, use_observed_distribution: bool) -> PathRung:
    tpm, stationary = macro_tpm_from_labels(labels)
    primitives = causal_primitives(
        tpm,
        intervention_distribution=stationary if use_observed_distribution else None,
    )
    return PathRung(
        primitives.n_states,
        primitives.cp,
        primitives.determinism,
        primitives.specificity,
    )


def _flatten_summary(
    coupling: float,
    seed: int,
    design: str,
    method_family: str,
    summary: dict[str, object],
) -> dict[str, object]:
    rungs: list[PathRung] = summary["rungs"]  # type: ignore[assignment]
    if rungs:
        cps = [rung.cp for rung in rungs]
        sizes = [rung.n_states for rung in rungs]
        cp_max = float(max(cps))
        cp_min = float(min(cps))
        cp_final = float(cps[-1])
        det_final = float(rungs[-1].determinism)
        spec_final = float(rungs[-1].specificity)
        max_size = int(max(sizes))
        min_size = int(min(sizes))
    else:
        cp_max = cp_min = cp_final = det_final = spec_final = float("nan")
        max_size = min_size = 0

    return {
        "coupling": coupling,
        "seed": seed,
        "obs_design": design,
        "method_family": method_family,
        "n_rungs": summary["n_rungs"],
        "max_n_states": max_size,
        "min_n_states": min_size,
        "cp_max": cp_max,
        "cp_min": cp_min,
        "cp_final": cp_final,
        "determinism_final": det_final,
        "specificity_final": spec_final,
        "total_ce": float(summary["total_ce"]),
        "emergent_complexity": float(summary["emergent_complexity"]),
        "emergent_complexity_raw_bits": float(summary.get("emergent_complexity_raw_bits", 0.0)),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_hierarchy(
    records: dict[tuple[float, int, str, str], dict[str, object]],
    design: str,
    outpath: Path,
) -> None:
    design_keys = [key for key in records if key[2] == design]
    if not design_keys:
        return

    method_families = sorted({key[3] for key in design_keys if not key[3].startswith("gt_")})
    couplings = sorted({key[0] for key in design_keys})
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        1,
        len(method_families),
        figsize=(3.4 * len(method_families), 3.4),
        sharey=True,
        squeeze=False,
    )
    for ax_idx, method in enumerate(method_families):
        ax = axes[0][ax_idx]
        for coupling_idx, coupling in enumerate(couplings):
            per_seed_curves: list[list[tuple[int, float]]] = []
            for key, summary in records.items():
                if key[2] != design or key[0] != coupling or key[3] != method:
                    continue
                rungs: list[PathRung] = summary["rungs"]  # type: ignore[assignment]
                if rungs:
                    per_seed_curves.append([(rung.n_states, rung.cp) for rung in rungs])
            if not per_seed_curves:
                continue

            sizes_union = sorted(
                {n_states for curve in per_seed_curves for n_states, _ in curve},
                reverse=True,
            )
            means: list[float] = []
            for n_states in sizes_union:
                values = [
                    cp
                    for curve in per_seed_curves
                    for size, cp in curve
                    if size == n_states
                ]
                means.append(float(np.mean(values)) if values else np.nan)
            ax.plot(
                sizes_union,
                means,
                marker="o",
                color=cmap(coupling_idx / max(len(couplings) - 1, 1)),
                label=f"ε={coupling:g}",
            )

        gt_cps: dict[float, list[float]] = defaultdict(list)
        for key, summary in records.items():
            if key[2] != design or key[3] != "gt_joint":
                continue
            rungs: list[PathRung] = summary["rungs"]  # type: ignore[assignment]
            if rungs:
                gt_cps[key[0]].append(rungs[0].cp)
        for coupling_idx, coupling in enumerate(couplings):
            values = gt_cps.get(coupling, [])
            if not values:
                continue
            ax.axhline(
                float(np.mean(values)),
                linestyle="--",
                color=cmap(coupling_idx / max(len(couplings) - 1, 1)),
                alpha=0.45,
                linewidth=1.0,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Macro dimensionality |M|")
        if ax_idx == 0:
            ax.set_ylabel("CP = determinism + specificity - 1")
        ax.set_title(method, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if ax_idx == len(method_families) - 1:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"CP along the coarsening path ({design} mixing)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ec(
    records: dict[tuple[float, int, str, str], dict[str, object]],
    design: str,
    outpath: Path,
) -> None:
    design_keys = [key for key in records if key[2] == design]
    if not design_keys:
        return

    method_families = sorted({key[3] for key in design_keys if not key[3].startswith("gt_")})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    cmap = plt.get_cmap("tab10")

    for method_idx, method in enumerate(method_families):
        ec_by_coupling: dict[float, list[float]] = defaultdict(list)
        total_ce_by_coupling: dict[float, list[float]] = defaultdict(list)
        for (coupling, _seed, obs_design, family), summary in records.items():
            if obs_design != design or family != method:
                continue
            ec_by_coupling[coupling].append(float(summary["emergent_complexity"]))
            total_ce_by_coupling[coupling].append(float(summary["total_ce"]))
        xs = sorted(ec_by_coupling.keys())
        if not xs:
            continue
        axes[0].plot(
            xs,
            [float(np.mean(total_ce_by_coupling[x])) for x in xs],
            marker="o",
            color=cmap(method_idx),
            label=method,
        )
        axes[1].plot(
            xs,
            [float(np.mean(ec_by_coupling[x])) for x in xs],
            marker="o",
            color=cmap(method_idx),
            label=method,
        )

    gt_ce_by_coupling: dict[float, list[float]] = defaultdict(list)
    for (coupling, _seed, obs_design, family), summary in records.items():
        if obs_design != design or family != "gt_joint":
            continue
        rungs: list[PathRung] = summary["rungs"]  # type: ignore[assignment]
        if rungs:
            gt_ce_by_coupling[coupling].append(rungs[0].cp)
    if gt_ce_by_coupling:
        xs = sorted(gt_ce_by_coupling.keys())
        axes[0].plot(
            xs,
            [float(np.mean(gt_ce_by_coupling[x])) for x in xs],
            marker="s",
            linestyle="--",
            color="black",
            label="gt_joint ceiling",
        )

    axes[0].set_xlabel("Coupling ε")
    axes[0].set_ylabel("Total CE = ΣΔCP⁺")
    axes[0].set_title(f"Total causal emergence ({design})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Coupling ε")
    axes[1].set_ylabel("Emergent complexity (normalised bits)")
    axes[1].set_title(f"Emergent complexity ({design})")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    root: Path,
    *,
    use_observed_distribution: bool = True,
) -> Path:
    csv_path = root / "recovery.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    recovery_rows = _read_recovery_csv(csv_path)
    builders = sorted(
        {
            row["method"].replace("iss_", "")
            for row in recovery_rows
            if row.get("method", "").startswith("iss_")
        }
    )
    npz_paths = _list_npz(root)
    if not npz_paths:
        raise FileNotFoundError(f"No labels_eps*.npz files under {root}")

    out_rows: list[dict[str, object]] = []
    path_records: dict[tuple[float, int, str, str], dict[str, object]] = {}

    for path in npz_paths:
        coupling, seed, design = _parse_npz_key(path)
        with np.load(path) as npz:
            target_len = max(npz[key].shape[0] for key in npz.files if npz[key].ndim >= 1)
            gt_paths = _ground_truth_path(npz, target_len=target_len)

            for builder in builders:
                chain = _path_for_builder(npz, builder)
                if chain:
                    chain = [_align_labels(labels, chain[0].shape[0]) for labels in chain]
                    summary = _summarise_path(
                        chain,
                        use_observed_distribution=use_observed_distribution,
                    )
                    key = (coupling, seed, design, f"iss_{builder}")
                    path_records[key] = summary
                    out_rows.append(_flatten_summary(coupling, seed, design, key[3], summary))

            pca_chain = _path_for_pca(npz)
            if pca_chain:
                pca_chain = [_align_labels(labels, pca_chain[0].shape[0]) for labels in pca_chain]
                summary = _summarise_path(
                    pca_chain,
                    use_observed_distribution=use_observed_distribution,
                )
                key = (coupling, seed, design, "pca_kmeans")
                path_records[key] = summary
                out_rows.append(_flatten_summary(coupling, seed, design, key[3], summary))

            for gt_name, labels in gt_paths.items():
                rung = _single_rung_cp(
                    labels,
                    use_observed_distribution=use_observed_distribution,
                )
                key = (coupling, seed, design, f"gt_{gt_name}")
                path_records[key] = {
                    "n_rungs": 1,
                    "rungs": [rung],
                    "delta_cp": [],
                    "total_ce": 0.0,
                    "emergent_complexity": 0.0,
                    "emergent_complexity_raw_bits": 0.0,
                }
                out_rows.append(
                    {
                        "coupling": coupling,
                        "seed": seed,
                        "obs_design": design,
                        "method_family": key[3],
                        "n_rungs": 1,
                        "max_n_states": rung.n_states,
                        "min_n_states": rung.n_states,
                        "cp_max": rung.cp,
                        "cp_min": rung.cp,
                        "cp_final": rung.cp,
                        "determinism_final": rung.determinism,
                        "specificity_final": rung.specificity,
                        "total_ce": 0.0,
                        "emergent_complexity": 0.0,
                        "emergent_complexity_raw_bits": 0.0,
                    }
                )

    out_csv = root / ("emergence.csv" if use_observed_distribution else "emergence_uniform.csv")
    _write_csv(out_csv, out_rows)

    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    designs = sorted({design for _, _, design, _ in path_records})
    for design in designs:
        hierarchy_name = (
            f"emergence_hierarchy_{design}.png"
            if use_observed_distribution
            else f"emergence_hierarchy_uniform_{design}.png"
        )
        ec_name = (
            f"emergence_ec_{design}.png"
            if use_observed_distribution
            else f"emergence_ec_uniform_{design}.png"
        )
        _plot_hierarchy(
            path_records,
            design,
            figures_dir / hierarchy_name,
        )
        _plot_ec(path_records, design, figures_dir / ec_name)

    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Directory containing recovery.csv and label archives")
    parser.add_argument(
        "--intervention-distribution",
        choices=("observed", "uniform"),
        default="observed",
        help="Use the empirical label distribution or a uniform intervention distribution for CP.",
    )
    args = parser.parse_args()
    out_csv = run_analysis(
        args.root,
        use_observed_distribution=args.intervention_distribution == "observed",
    )
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
