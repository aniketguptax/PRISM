import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_summary_by_condition(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def filter_rows(rows: List[Dict[str, str]], base_process: str, flip_p: float, subsample_step: int) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        if r.get("base_process") != base_process:
            continue
        if abs(float(r.get("flip_p", "0.0")) - flip_p) > 1e-12:
            continue
        if int(float(r.get("subsample_step", "1"))) != subsample_step:
            continue
        out.append(r)
    return out


def rows_to_xy(rows: List[Dict[str, str]], metric: str) -> Tuple[List[int], List[float], List[float]]:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    rows = sorted(rows, key=lambda r: int(float(r["k"])))
    ks = [int(float(r["k"])) for r in rows]
    ys = [float(r[mean_col]) for r in rows]
    yerr = [float(r.get(std_col, "0") or 0.0) for r in rows]
    return ks, ys, yerr


def plot_even_curves(rows: List[Dict[str, str]], outpath: Path) -> None:
    # 3 curves on one figure (clean + readable)
    fig = plt.figure(figsize=(6.4, 3.4))

    # logloss (nats)
    ks, ys, yerr = rows_to_xy(rows, "logloss")
    plt.errorbar(ks, ys, yerr=yerr, marker="o", capsize=3, label="log-loss")

    # C_mu_empirical (nats)
    ks2, ys2, yerr2 = rows_to_xy(rows, "C_mu_empirical")
    plt.errorbar(ks2, ys2, yerr=yerr2, marker="o", capsize=3, label=r"$C_{\mu}^{emp}$ (nats")

    # branching entropy (bits)
    ks3, ys3, yerr3 = rows_to_xy(rows, "branch_entropy")
    plt.errorbar(ks3, ys3, yerr=yerr3, marker="o", capsize=3, label=r"$H_{\mathrm{br}}$ (bits)")

    plt.xlabel("Representation length $k$")
    plt.ylabel("Metric value (see legend)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close(fig)


def plot_finite_data_compare(
    rows_Tsmall: List[Dict[str, str]],
    rows_Tlarge: List[Dict[str, str]],
    k_fixed: int,
    outpath: Path
) -> None:
    def pick_k(rows: List[Dict[str, str]], k: int) -> Dict[str, str]:
        for r in rows:
            if int(float(r["k"])) == k:
                return r
        raise ValueError(f"Missing k={k} in rows")

    rS = pick_k(rows_Tsmall, k_fixed)
    rL = pick_k(rows_Tlarge, k_fixed)

    fig = plt.figure(figsize=(5.6, 3.2))

    # We’ll compare two metrics with errorbars (across seeds): Cmu and Hbr
    labels = [f"T={int(float(rS['n']))} runs (small)", f"T={int(float(rL['n']))} runs (large)"]  # not perfect but harmless

    metrics = [
        ("C_mu_empirical", r"$C_{\mu}^{emp}$ (nats)"),
        ("branch_entropy", r"$H_{\mathrm{br}}$ (bits)"),
    ]

    x = [0, 1]
    for j, (m, title) in enumerate(metrics):
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        yS = float(rS[mean_col]); eS = float(rS.get(std_col, "0") or 0.0)
        yL = float(rL[mean_col]); eL = float(rL.get(std_col, "0") or 0.0)

        # slight horizontal offset per metric so both appear
        dx = (-0.08 if j == 0 else 0.08)
        plt.errorbar([x[0] + dx, x[1] + dx], [yS, yL], yerr=[eS, eL], marker="o", capsize=3, label=title)

    plt.xticks(x, ["small T", "large T"])
    plt.ylabel("Value (mean ± s.d. across seeds)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--even-root", type=Path, required=True, help="Even Process sweep root (has summary_by_condition.csv)")
    p.add_argument("--even-base-process", type=str, default="even_process")
    p.add_argument("--flip-p", type=float, default=0.0)
    p.add_argument("--subsample-step", type=int, default=1)

    p.add_argument("--finite-small-root", type=Path, required=True)
    p.add_argument("--finite-large-root", type=Path, required=True)
    p.add_argument("--finite-k", type=int, default=8)

    p.add_argument("--outdir", type=Path, required=True)

    args = p.parse_args()

    even_rows = read_summary_by_condition(args.even_root / "summary_by_condition.csv")
    even_rows = filter_rows(even_rows, args.even_base_process, args.flip_p, args.subsample_step)

    rows_small = read_summary_by_condition(args.finite_small_root / "summary_by_condition.csv")
    rows_small = filter_rows(rows_small, args.even_base_process, args.flip_p, args.subsample_step)

    rows_large = read_summary_by_condition(args.finite_large_root / "summary_by_condition.csv")
    rows_large = filter_rows(rows_large, args.even_base_process, args.flip_p, args.subsample_step)

    plot_even_curves(even_rows, args.outdir / "even_emergence_curves_vs_k.pdf")
    plot_finite_data_compare(rows_small, rows_large, args.finite_k, args.outdir / "even_finite_data_compare_k8.pdf")

    print("Wrote:")
    print(" -", args.outdir / "even_emergence_curves_vs_k.pdf")
    print(" -", args.outdir / "even_finite_data_compare_k8.pdf")


if __name__ == "__main__":
    main()