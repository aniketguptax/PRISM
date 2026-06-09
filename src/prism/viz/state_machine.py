from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch

Edge = tuple[int, int, int, float]  # (state, symbol, next_state, probability)

INK = "#1F2937"
MUTED = "#6B7280"
NODE_EDGE = "#1F2937"
LABEL_FILL = "#FFFFFF"
LABEL_EDGE = "#D6DCE5"

SYMBOL_COLOURS: dict[int, str] = {
    0: "#2F5D8C",
    1: "#B0413E", 
}
SYMBOL_COLOUR_FALLBACK = "#3F4A5A"

EMISSION_CMAP = LinearSegmentedColormap.from_list(
    "prism_emission",
    [
        (0.00, "#2F5D8C"),
        (0.46, "#9DBAD3"),
        (0.50, "#ECEFF3"),
        (0.54, "#E2B6B1"),
        (1.00, "#B0413E"),
    ],
)
EMISSION_NORM = Normalize(vmin=0.0, vmax=1.0)

@dataclass(frozen=True)
class MachineLayout:
    positions: dict[int, tuple[float, float]] | None = None
    loops: dict[int, str] = field(default_factory=dict)
    curve: dict[tuple[int, int], float] = field(default_factory=dict)
    edge_label_t: dict[tuple[int, int, int], float] = field(default_factory=dict)
    edge_label_offsets: dict[tuple[int, int, int], tuple[float, float]] = field(
        default_factory=dict
    )
    figsize: tuple[float, float] | None = None
    limits: tuple[float, float, float, float] | None = None

@dataclass(frozen=True)
class StateMachineFigure:
    edges: tuple[Edge, ...]
    emission: dict[int, float] = field(default_factory=dict)
    occupancy: dict[int, float] = field(default_factory=dict)
    layout: MachineLayout = field(default_factory=MachineLayout)
    title: str | None = None
    show_colorbar: bool = True

def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.0,
        }
    )

def _circular_positions(states: Sequence[int]) -> dict[int, tuple[float, float]]:
    n = len(states)
    if n == 0:
        return {}
    if n == 1:
        return {states[0]: (0.0, 0.0)}
    if n == 2:
        return {states[0]: (-0.95, 0.0), states[1]: (0.95, 0.0)}
    radius = 1.0 + 0.06 * n
    out: dict[int, tuple[float, float]] = {}
    for i, s in enumerate(states):
        theta = math.pi / 2.0 - 2.0 * math.pi * i / n
        out[s] = (radius * math.cos(theta), radius * math.sin(theta))
    return out


def _auto_limits(
    positions: Mapping[int, tuple[float, float]], pad: float
) -> tuple[float, float, float, float]:
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)


def _auto_figsize(limits: tuple[float, float, float, float]) -> tuple[float, float]:
    left, right, bottom, top = limits
    w = right - left
    h = top - bottom
    scale = 1.15
    return (max(2.0, w * scale), max(1.9, h * scale))


def _loop_side(
    pos: tuple[float, float], centroid: tuple[float, float]
) -> str:
    dx = pos[0] - centroid[0]
    dy = pos[1] - centroid[1]
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "up" if dy >= 0 else "down"


def _curve_point(
    start: tuple[float, float],
    end: tuple[float, float],
    rad: float,
    t: float,
) -> tuple[float, float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    x = start[0] + t * dx
    y = start[1] + t * dy
    if dist == 0.0 or rad == 0.0:
        return x, y
    nx = -dy / dist
    ny = dx / dist
    bow = 4.0 * t * (1.0 - t)
    return x + 0.5 * rad * dist * bow * nx, y + 0.5 * rad * dist * bow * ny

def _relative_luminance(colour) -> float:
    r, g, b = to_rgb(colour)

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

NODE_RADIUS = 0.27

def _draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
    colour: str = INK,
    lw: float = 1.05,
    shrink: float = 23.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=colour,
            shrinkA=shrink,
            shrinkB=shrink,
            connectionstyle=f"arc3,rad={rad}",
            capstyle="round",
            joinstyle="round",
            zorder=2,
        )
    )


def _label_card(ax: Axes, xy: tuple[float, float], text: str, colour: str) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        color=colour,
        linespacing=1.15,
        bbox={
            "boxstyle": "round,pad=0.18,rounding_size=0.12",
            "fc": LABEL_FILL,
            "ec": LABEL_EDGE,
            "lw": 0.5,
            "alpha": 0.97,
        },
        zorder=6,
    )


def _edge_label(ax: Axes, xy: tuple[float, float], symbol: int, prob: float) -> None:
    colour = SYMBOL_COLOURS.get(symbol, SYMBOL_COLOUR_FALLBACK)
    _label_card(ax, xy, _format_symbol(symbol, prob), colour)


def _format_symbol(symbol: int, prob: float) -> str:
    if abs(prob - 1.0) < 1e-9:
        return rf"$\mathbf{{{symbol}}}$"
    return rf"$\mathbf{{{symbol}}}\!:\!{prob:.2f}$"


def _draw_loop(
    ax: Axes,
    pos: tuple[float, float],
    symps: Sequence[tuple[int, float]],
    *,
    side: str,
) -> None:
    x, y = pos
    symbols = sorted(symps, key=lambda t: t[0])
    distinct = {sym for sym, _ in symbols}
    colour = (
        SYMBOL_COLOURS.get(next(iter(distinct)), SYMBOL_COLOUR_FALLBACK)
        if len(distinct) == 1
        else SYMBOL_COLOUR_FALLBACK
    )
    label_text = "\n".join(_format_symbol(sym, prob) for sym, prob in symbols)
    r = NODE_RADIUS
    if side == "down":
        start = (x + 0.86 * r, y - 0.66 * r)
        end = (x - 0.86 * r, y - 0.66 * r)
        rad = -1.7
        label_xy = (x, y - 0.74)
    elif side == "left":
        start = (x - 0.66 * r, y + 0.86 * r)
        end = (x - 0.66 * r, y - 0.86 * r)
        rad = 1.7
        label_xy = (x - 0.78, y)
    elif side == "right":
        start = (x + 0.66 * r, y - 0.86 * r)
        end = (x + 0.66 * r, y + 0.86 * r)
        rad = 1.7
        label_xy = (x + 0.78, y)
    else:
        start = (x - 0.86 * r, y + 0.66 * r)
        end = (x + 0.86 * r, y + 0.66 * r)
        rad = -1.7
        label_xy = (x, y + 0.74)
    _draw_arrow(ax, start, end, rad=rad, shrink=3.0, lw=1.0, colour=colour)
    _label_card(ax, label_xy, label_text, colour)


def _node_fill(state: int, emission: Mapping[int, float]) -> tuple:
    if state in emission:
        return EMISSION_CMAP(EMISSION_NORM(float(emission[state])))
    return to_rgb("#EEF2F7")


def _draw_nodes(
    ax: Axes,
    positions: Mapping[int, tuple[float, float]],
    emission: Mapping[int, float],
) -> None:
    for state, (x, y) in positions.items():
        fill = _node_fill(state, emission)
        ax.add_patch(
            Circle(
                (x, y),
                radius=NODE_RADIUS + 0.018,
                facecolor="none",
                edgecolor="white",
                linewidth=2.4,
                zorder=3,
            )
        )
        ax.add_patch(
            Circle(
                (x, y),
                radius=NODE_RADIUS,
                facecolor=fill,
                edgecolor=NODE_EDGE,
                linewidth=1.05,
                zorder=4,
            )
        )
        txt_colour = "white" if _relative_luminance(fill) < 0.5 else INK
        ax.text(
            x,
            y,
            rf"$S_{{{state}}}$",
            ha="center",
            va="center",
            fontsize=11.0,
            color=txt_colour,
            zorder=5,
        )


def _draw_colorbar(fig, ax: Axes) -> None:
    cax = ax.inset_axes((0.0, -0.12, 1.0, 0.05))
    sm = mpl.cm.ScalarMappable(norm=EMISSION_NORM, cmap=EMISSION_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(LABEL_EDGE)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["0", "0.5", "1"])
    cax.tick_params(length=2.0, width=0.5, labelsize=6.6, colors=MUTED, pad=1.5)
    cb.set_label(
        r"predictive emission  $\hat p(x_{t+1}{=}1 \mid \mathrm{state})$",
        fontsize=6.9,
        color=INK,
        labelpad=2.5,
    )


def _reciprocal_pairs(edges: Iterable[Edge]) -> set[tuple[int, int]]:
    directed = {(s, sp) for s, _, sp, _ in edges if s != sp}
    out: set[tuple[int, int]] = set()
    for (u, v) in directed:
        if (v, u) in directed:
            out.add((u, v))
    return out


def render_state_machine(
    spec: StateMachineFigure,
    out_stem: Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 320,
) -> Path:
    edges = list(spec.edges)
    states = sorted({s for s, _, _, _ in edges} | {sp for _, _, sp, _ in edges})

    layout = spec.layout
    positions = layout.positions or _circular_positions(states)
    centroid = (
        sum(p[0] for p in positions.values()) / max(len(positions), 1),
        sum(p[1] for p in positions.values()) / max(len(positions), 1),
    )

    limits = layout.limits or _auto_limits(positions, pad=0.85)
    figsize = layout.figsize or _auto_figsize(limits)

    _set_style()
    fig, ax = plt.subplots(figsize=figsize)

    reciprocal = _reciprocal_pairs(edges)

    # Non-self transitions
    for s, sym, sp, prob in edges:
        if s == sp:
            continue
        start = positions[s]
        end = positions[sp]
        if (s, sp) in layout.curve:
            rad = layout.curve[(s, sp)]
        elif (s, sp) in reciprocal:
            rad = 0.18
        else:
            rad = 0.0
        _draw_arrow(ax, start, end, rad=rad, colour=SYMBOL_COLOURS.get(sym, SYMBOL_COLOUR_FALLBACK))

        t = layout.edge_label_t.get((s, sym, sp), 0.5)
        lx, ly = _curve_point(start, end, rad, t)
        dx, dy = layout.edge_label_offsets.get((s, sym, sp), (0.0, 0.0))
        _edge_label(ax, (lx + dx, ly + dy), sym, prob)

    # Self-loops
    self_loops: dict[int, list[tuple[int, float]]] = {}
    for s, sym, sp, prob in edges:
        if s == sp:
            self_loops.setdefault(s, []).append((sym, prob))
    for s, symps in self_loops.items():
        side = layout.loops.get(s) or _loop_side(positions[s], centroid)
        _draw_loop(ax, positions[s], symps, side=side)

    _draw_nodes(ax, positions, spec.emission)

    left, right, bottom, top = limits
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.axis("off")

    if spec.title:
        ax.set_title(spec.title, fontsize=9.5, fontweight="bold", color=INK, pad=6.0)

    if spec.show_colorbar and spec.emission:
        _draw_colorbar(fig, ax)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    first: Path | None = None
    for fmt in formats:
        out = out_stem.with_suffix(f".{fmt}")
        save_kwargs = {"dpi": dpi} if fmt != "pdf" else {}
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04, **save_kwargs)
        if first is None:
            first = out
    plt.close(fig)
    return first if first is not None else out_stem


def render_from_edges(
    edges: Iterable[Edge],
    out_stem: Path,
    *,
    emission: Mapping[int, float] | None = None,
    occupancy: Mapping[int, float] | None = None,
    title: str | None = None,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 320,
) -> Path:
    spec = StateMachineFigure(
        edges=tuple((int(s), int(sym), int(sp), float(p)) for s, sym, sp, p in edges),
        emission=dict(emission or {}),
        occupancy=dict(occupancy or {}),
        title=title,
    )
    return render_state_machine(spec, out_stem, formats=formats, dpi=dpi)
