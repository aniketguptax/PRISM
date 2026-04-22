"""Generate the block-modular setup figure used in the chapter draft."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prism.processes.block_modular_lgssm import BlockModularLGSSM


def _sample_latent(coupling: float, seed: int, length: int) -> np.ndarray:
    process = BlockModularLGSSM(coupling=coupling, obs_dim=8, obs_design="random")
    sample = process.sample(length=length, seed=seed)
    return np.asarray(sample.latent, dtype=float)


def _spectrum(coupling: float, seed: int) -> np.ndarray:
    process = BlockModularLGSSM(coupling=coupling, obs_dim=8, obs_design="random")
    rng = np.random.default_rng(seed)
    return np.linalg.eigvals(process._build_A(rng))


def run(outpath: Path, seed: int = 0, length: int = 400) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))

    latent = _sample_latent(coupling=0.0, seed=seed, length=length)
    t = np.arange(length)
    axes[0].plot(t, latent[:, 0], label=r"$Z^{(1)}_1$", color="#1f77b4", linewidth=1.0)
    axes[0].plot(t, latent[:, 1], label=r"$Z^{(1)}_2$", color="#6baed6", linewidth=1.0)
    axes[0].plot(t, latent[:, 2], label=r"$Z^{(2)}_1$", color="#d62728", linewidth=1.0)
    axes[0].plot(t, latent[:, 3], label=r"$Z^{(2)}_2$", color="#ff9896", linewidth=1.0)
    axes[0].set_title(r"(a) Latent trajectories, $\varepsilon=0$", fontsize=10)
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Latent value")
    axes[0].legend(fontsize=7, loc="upper right", ncols=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(latent[:, 0], latent[:, 1], color="#1f77b4", alpha=0.75, linewidth=0.8, label="slow block")
    axes[1].plot(latent[:, 2], latent[:, 3], color="#d62728", alpha=0.75, linewidth=0.8, label="rotational block")
    axes[1].set_title(r"(b) Phase portraits, $\varepsilon=0$", fontsize=10)
    axes[1].set_xlabel("First coordinate")
    axes[1].set_ylabel("Second coordinate")
    axes[1].legend(fontsize=8, loc="best")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].grid(True, alpha=0.3)

    couplings = [0.0, 0.05, 0.10, 0.20]
    markers = ["o", "s", "^", "D"]
    colours = ["#000000", "#2ca02c", "#ff7f0e", "#9467bd"]
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    axes[2].plot(np.cos(theta), np.sin(theta), color="lightgray", linestyle="--", linewidth=0.8)
    for coupling, marker, colour in zip(couplings, markers, colours):
        eigenvalues = _spectrum(coupling=coupling, seed=seed)
        axes[2].scatter(
            eigenvalues.real,
            eigenvalues.imag,
            marker=marker,
            s=40,
            facecolor="none",
            edgecolor=colour,
            linewidth=1.2,
            label=rf"$\varepsilon={coupling:g}$",
        )
    axes[2].set_title(r"(c) Eigenvalues of $A$", fontsize=10)
    axes[2].set_xlabel(r"Re$(\lambda)$")
    axes[2].set_ylabel(r"Im$(\lambda)$")
    axes[2].set_xlim(-1.05, 1.05)
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend(fontsize=7, loc="lower left")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=400)
    args = parser.parse_args()
    run(outpath=args.out, seed=args.seed, length=args.length)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
