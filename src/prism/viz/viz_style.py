from __future__ import annotations

import matplotlib as mpl
from matplotlib.axes import Axes

INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E8EDF2"
BLUE = "#2F5D8C"
RED = "#B0413E"
PURPLE = "#6F55A4"
GREEN = "#2F7F7F"
GOLD = "#C8881F"
TEAL = "#2F7F7F"
SLATE = "#4F5B6B"

SERIES = (BLUE, RED, GREEN, PURPLE, GOLD, SLATE)

def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#2F3742",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 8.6,
            "axes.titleweight": "bold",
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axes(ax: Axes, *, grid: bool = True, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis=grid_axis, color=GRID, lw=0.7)
        ax.set_axisbelow(True)


def panel_label(ax: Axes, label: str, *, x: float = -0.18, y: float = 1.12) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


def title(ax: Axes, text: str, *, pad: float = 6.0) -> None:
    ax.set_title(text, fontsize=8.6, fontweight="bold", pad=pad, color=INK)


def line(
    ax: Axes,
    x,
    y,
    *,
    colour: str,
    label: str | None = None,
    lw: float = 1.9,
    ms: float = 4.0,
    marker: str = "o",
    ls: str = "-",
    zorder: int = 3,
):
    return ax.plot(
        x,
        y,
        marker=marker,
        ms=ms,
        lw=lw,
        ls=ls,
        color=colour,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=label,
        zorder=zorder,
    )


def band(ax: Axes, x, lo, hi, *, colour: str, alpha: float = 0.15) -> None:
    ax.fill_between(x, lo, hi, color=colour, alpha=alpha, linewidth=0)


def legend(ax: Axes, **kwargs) -> None:
    defaults = dict(frameon=False, fontsize=7.0, handlelength=1.6)
    defaults.update(kwargs)
    ax.legend(**defaults)


def savefig(fig, stem, *, dpi: int = 320) -> None:
    from pathlib import Path

    stem = Path(stem)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)
