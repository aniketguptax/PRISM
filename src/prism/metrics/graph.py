from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import subprocess

from prism.reconstruction.protocols import TransitionModel

Edge = Tuple[int, int, int, float]  # (state, symbol, next_state, probability)


@dataclass(frozen=True)
class DotStyle:
    rankdir: str = "LR"
    splines: str = "true"
    overlap: str = "false"
    concentrate: str = "false"
    outputorder: str = "edgesfirst"
    nodesep: float = 1.3
    ranksep: float = 1.6
    pad: float = 0.45

    # Extra spacing around nodes/edges so labels can settle clear of splines
    # (graphviz uses these as soft padding when routing).
    sep: str = "+18,18"
    esep: str = "+18,18"

    fontname: str = "Helvetica"
    font_colour: str = "#1f2933"
    graph_fontsize: int = 18
    node_fontsize: int = 22
    edge_fontsize: int = 13

    node_shape: str = "ellipse"
    node_fixedsize: bool = True
    node_width: float = 0.95
    node_height: float = 0.75
    node_penwidth: float = 1.5
    node_margin: float = 0.10
    node_colour: str = "#1f2933"
    node_font_colour: Optional[str] = None
    node_style: str = "filled"
    node_fillcolour: str = "#eef2f7"

    arrowsize: float = 0.9
    edge_penwidth: float = 1.2
    edge_colour: str = "#374151"
    edge_font_colour: Optional[str] = None

    # When false, graphviz tries to keep edge labels from sitting on top of
    # other edges (at the cost of wider layouts). True allows looser placement.
    labelfloat: str = "false"
    labeldistance: float = 2.6
    labelangle: float = 28.0

    # reciprocal edges: push labels further out + stronger angle
    reciprocal_labeldistance: float = 3.1
    reciprocal_labelangle: float = 36.0
    reciprocal_minlen: int = 3
    reciprocal_weight: int = 2

    selfloop_minlen: int = 2

    # All transition probabilities render at fixed 3-dp precision (including
    # 0.000 and 1.000) so deterministic and stochastic edges read consistently.
    prob_precision: int = 3
    strip_trailing_zeros: bool = False
    clamp_probabilities: bool = True
    prob_epsilon: float = 1e-12
    # Always emit "<sym>: <prob>" so deterministic transitions still show the
    # 1.000 (Nature-style consistency); set True for symbol-only collapse.
    hide_unit_probability: bool = False

    state_prefix: str = "S"
    # Render state labels with an italic letter and subscript index (S_0 style).
    italic_state_subscript: bool = True

    # If True: use xlabel (external, floating) — labels can land anywhere and
    # may sit on top of unrelated splines. If False: use inline `label`, so
    # graphviz splits the owning edge spline around the label and other edges
    # have to route clear of it. False is the publication-ready choice.
    use_external_edge_labels: bool = False

    # Use HTML table labels. Backgrounds are kept transparent so overlapping
    # splines stay visible underneath instead of being chopped by white boxes.
    use_html_labels: bool = True
    label_cellpadding: int = 2
    label_cellspacing: int = 0
    label_cellborder: int = 0
    label_symbol_bold: bool = True
    # "none" => transparent label background; any colour string => filled card.
    label_bgcolour: str = "none"


def to_edge_list(model: TransitionModel) -> List[Edge]:
    edges: List[Edge] = []
    for (s, sym), sp_dict in model.transitions.items():
        for sp, prob in sp_dict.items():
            edges.append((s, sym, sp, float(prob)))
    return edges


def _escape_dot(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_prob(p: float, precision: int, strip_trailing_zeros: bool) -> str:
    text = f"{p:.{precision}f}"
    if strip_trailing_zeros:
        text = text.rstrip("0").rstrip(".")
        if text == "-0":
            text = "0"
    return text


def _clean_prob(p: float, clamp: bool, eps: float) -> Optional[float]:
    if p is None:
        return None
    if isinstance(p, float) and (math.isnan(p) or math.isinf(p)):
        return None
    v = float(p)
    if clamp:
        if abs(v) < eps:
            v = 0.0
        elif abs(v - 1.0) < eps:
            v = 1.0
        elif -eps <= v < 0.0:
            v = 0.0
        elif 1.0 < v <= 1.0 + eps:
            v = 1.0
    return v


def _ports_for_rankdir(rankdir: str) -> Tuple[str, str]:
    rd = (rankdir or "").upper()
    if rd in ("LR", "RL"):
        return "e", "w"
    if rd in ("TB", "BT"):
        return "s", "n"
    return "e", "w"


def to_dot(
    edges: Iterable[Edge],
    graph_name: str = "state_machine",
    label: Optional[str] = None,
    style: Optional[DotStyle] = None,
    symbol_names: Optional[Dict[int, str]] = None,
    state_names: Optional[Dict[int, str]] = None,
    merge_parallel: bool = True,
    sort_symbols: bool = True,
) -> str:
    st = style or DotStyle()
    edges_list = list(edges)

    nodes = sorted({s for s, _, _, _ in edges_list} | {sp for _, _, sp, _ in edges_list})
    if not nodes:
        gname = _escape_dot(graph_name)
        return f'digraph "{gname}" {{}}'

    directed_pairs: Set[Tuple[int, int]] = {(s, sp) for s, _, sp, _ in edges_list if s != sp}
    reciprocal_undirected: Set[Tuple[int, int]] = set()
    for (u, v) in directed_pairs:
        if (v, u) in directed_pairs:
            a, b = (u, v) if u < v else (v, u)
            reciprocal_undirected.add((a, b))

    grouped: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
    if merge_parallel:
        for s, sym, sp, p in edges_list:
            grouped.setdefault((s, sp), []).append((sym, float(p)))
    else:
        for s, sym, sp, p in edges_list:
            grouped.setdefault((s * 10**9 + sym, sp), []).append((sym, float(p)))

    def sym_name(sym: int) -> str:
        if symbol_names and sym in symbol_names:
            return str(symbol_names[sym])
        return str(sym)

    def state_name(state: int) -> str:
        if state_names and state in state_names:
            return str(state_names[state])
        return f"{st.state_prefix}{state}" if st.state_prefix else str(state)

    def state_label_html(state: int) -> Optional[str]:
        """HTML state label as an italic letter with a subscript index, e.g. <I>S</I><SUB>0</SUB>."""
        if not st.italic_state_subscript or not st.state_prefix:
            return None
        if state_names and state in state_names:
            return None
        return f"<<I>{_escape_html(st.state_prefix)}</I><SUB>{state}</SUB>>"

    tail_base, head_base = _ports_for_rankdir(st.rankdir)
    is_lr = st.rankdir.upper() in ("LR", "RL")

    lines: List[str] = []
    gname = _escape_dot(graph_name)

    lines.append(f'digraph "{gname}" {{')
    lines.append(f"  rankdir={st.rankdir};")
    lines.append(f"  splines={st.splines};")
    lines.append(f"  overlap={st.overlap};")
    lines.append(f"  concentrate={st.concentrate};")
    lines.append(f"  outputorder={st.outputorder};")
    lines.append(f"  nodesep={st.nodesep};")
    lines.append(f"  ranksep={st.ranksep};")
    lines.append(f"  pad={st.pad};")
    lines.append(f'  sep="{st.sep}";')
    lines.append(f'  esep="{st.esep}";')

    if st.use_external_edge_labels:
        lines.append("  forcelabels=true;")

    lines.append(
        f'  graph [fontname="{st.fontname}", fontsize={st.graph_fontsize}, fontcolor="{st.font_colour}"];'
    )

    node_attrs = [
        f"shape={st.node_shape}",
        f'fontname="{st.fontname}"',
        f"fontsize={st.node_fontsize}",
        f"penwidth={st.node_penwidth}",
        f'color="{st.node_colour}"',
        f'fontcolor="{st.node_font_colour or st.font_colour}"',
        f"margin={st.node_margin}",
    ]
    if st.node_style:
        node_attrs.append(f'style="{st.node_style}"')
        node_attrs.append(f'fillcolor="{st.node_fillcolour}"')
    if st.node_fixedsize:
        node_attrs += ["fixedsize=true", f"width={st.node_width}", f"height={st.node_height}"]
    lines.append(f"  node [{', '.join(node_attrs)}];")

    edge_attrs = [
        f'fontname="{st.fontname}"',
        f"fontsize={st.edge_fontsize}",
        f"arrowsize={st.arrowsize}",
        f"penwidth={st.edge_penwidth}",
        f'color="{st.edge_colour}"',
        f'fontcolor="{st.edge_font_colour or st.font_colour}"',
        "decorate=false",
        f"labelfloat={st.labelfloat}",
    ]
    lines.append(f"  edge [{', '.join(edge_attrs)}];")

    if label is not None:
        safe_label = _escape_dot(label)
        lines.append(f'  label="{safe_label}";')
        lines.append("  labelloc=t;")
        lines.append("  labeljust=l;")

    for n in nodes:
        html_label = state_label_html(n)
        if html_label is not None:
            lines.append(f"  {n} [label={html_label}];")
        else:
            lines.append(f'  {n} [label="{_escape_dot(state_name(n))}"];')

    def build_label(symps: List[Tuple[int, float]]) -> Tuple[Optional[str], bool]:
        rows_out: List[Tuple[str, Optional[str]]] = []
        for sym, p in symps:
            cp = _clean_prob(p, st.clamp_probabilities, st.prob_epsilon)
            if cp is None:
                continue
            if st.hide_unit_probability and cp == 1.0:
                rows_out.append((sym_name(sym), None))
            else:
                prob_text = _format_prob(cp, st.prob_precision, st.strip_trailing_zeros)
                rows_out.append((sym_name(sym), prob_text))

        if not rows_out:
            return None, False

        if st.use_html_labels:
            rows: List[str] = []
            for sym_text, prob_text in rows_out:
                s_esc = _escape_html(sym_text)
                if st.label_symbol_bold:
                    s_esc = f"<B>{s_esc}</B>"
                if prob_text is None:
                    rows.append(
                        "<TR>"
                        f'<TD ALIGN="CENTER" COLSPAN="2">{s_esc}</TD>'
                        "</TR>"
                    )
                else:
                    rows.append(
                        "<TR>"
                        f'<TD ALIGN="RIGHT">{s_esc}:</TD>'
                        f'<TD ALIGN="LEFT">{_escape_html(prob_text)}</TD>'
                        "</TR>"
                    )
            bg_attr = (
                f' BGCOLOR="{st.label_bgcolour}"'
                if st.label_bgcolour and st.label_bgcolour.lower() != "none"
                else ""
            )
            table = (
                f'<TABLE BORDER="0" CELLBORDER="{st.label_cellborder}" '
                f'CELLSPACING="{st.label_cellspacing}" '
                f'CELLPADDING="{st.label_cellpadding}"'
                f"{bg_attr}>"
                f'{"".join(rows)}'
                f"</TABLE>"
            )
            return f"<{table}>", True

        text = "\\n".join(s if p is None else f"{s}: {p}" for s, p in rows_out)
        return _escape_dot(text), False

    def _pmax(symps: List[Tuple[int, float]]) -> float:
        best = 0.0
        for _, p in symps:
            cp = _clean_prob(p, st.clamp_probabilities, st.prob_epsilon)
            if cp is None:
                continue
            if cp > best:
                best = cp
        return best

    for (s, sp), symps in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if sort_symbols:
            symps = sorted(symps, key=lambda t: t[0])

        label_text, label_is_html = build_label(symps)
        if label_text is None:
            continue

        # Self-loop
        if s == sp:
            attrs: List[str] = [f"minlen={st.selfloop_minlen}"]
            if label_is_html:
                attrs.append(f"label={label_text}")
            else:
                attrs.append(f'label="{label_text}"')
            lines.append(f"  {s} -> {sp} [{', '.join(attrs)}];")
            continue

        # Reciprocal separation: split parallel edges with `dir=both`-friendly
        # routing instead of forcing compass ports. Forced ports (e.g. ne/nw)
        # only render correctly if graphviz actually places s to the left of sp;
        # for back-edges that are not the case, so the arrowhead can land on
        # the wrong end. Letting graphviz choose the ports keeps the direction
        # honest at the cost of slightly less symmetric routing.
        a, b = (s, sp) if s < sp else (sp, s)
        is_recip = (a, b) in reciprocal_undirected
        if is_recip:
            attrs = [
                f"minlen={st.reciprocal_minlen}",
                f"weight={st.reciprocal_weight}",
                f"labeldistance={st.reciprocal_labeldistance}",
            ]

            if st.use_external_edge_labels:
                attrs.append('label=""')
                if label_is_html:
                    attrs.append(f"xlabel={label_text}")
                else:
                    attrs.append(f'xlabel="{label_text}"')
            else:
                if label_is_html:
                    attrs.append(f"label={label_text}")
                else:
                    attrs.append(f'label="{label_text}"')

            lines.append(f"  {s} -> {sp} [{', '.join(attrs)}];")
            continue

        # Normal directed edge — let graphviz choose ports so arrowheads always
        # land at sp regardless of left/right placement.
        attrs = []

        # Keep edges visible: give dot routing room; boost “important” edges.
        pm = _pmax(symps)
        if pm >= 0.5:
            attrs += ["minlen=2", "weight=3"]
        else:
            attrs += ["minlen=2", "weight=2"]

        if st.use_external_edge_labels:
            attrs.append('label=""')
            if label_is_html:
                attrs.append(f"xlabel={label_text}")
            else:
                attrs.append(f'xlabel="{label_text}"')

            # Push labels consistently away from the spline.
            attrs.append(f"labeldistance={st.labeldistance}")
            if is_lr:
                attrs.append("labelangle=90")
            else:
                attrs.append("labelangle=0")
        else:
            attrs.append(f"labeldistance={st.labeldistance}")
            attrs.append(f"labelangle={st.labelangle}")
            if label_is_html:
                attrs.append(f"label={label_text}")
            else:
                attrs.append(f'label="{label_text}"')

        lines.append(f"  {s} -> {sp} [{', '.join(attrs)}];")

    lines.append("}")
    return "\n".join(lines)


def save_dot(path: Path, dot: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dot, encoding="utf-8")


def dot_to_png(dot_path: Path, png_path: Path, dpi: int = 600) -> None:
    subprocess.run(
        ["dot", f"-Gdpi={dpi}", "-Tpng", str(dot_path), "-o", str(png_path)],
        check=True,
    )
