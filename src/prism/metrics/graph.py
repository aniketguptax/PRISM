from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Iterable, Optional

from prism.reconstruction.protocols import PredictiveStateModel


Edge = Tuple[int, int, int]  # (state, symbol, next_state)


def to_edge_list(model: PredictiveStateModel) -> List[Edge]:
    return [(s, sym, sp) for (s, sym), sp in model.transitions.items()]


def to_dot(
    edges: Iterable[Edge],
    graph_name: str = "state_machine",
    rankdir: str = "LR",
    label: Optional[str] = None,
) -> str:
    edges = list(edges)
    nodes = sorted({s for s, _, _ in edges} | {sp for _, _, sp in edges})

    lines = []
    lines.append(f'digraph "{graph_name}" {{')
    lines.append(f"  rankdir={rankdir};")
    lines.append("  node [shape=circle];")

    if label is not None:
        safe_label = label.replace('"', '\\"')
        lines.append(f'  label="{safe_label}";')
        lines.append("  labelloc=t;")

    for n in nodes:
        lines.append(f'  {n} [label="{n}"];')

    for s, sym, sp in sorted(edges, key=lambda e: (e[0], e[1], e[2])):
        lines.append(f'  {s} -> {sp} [label="{sym}"];')

    lines.append("}")
    return "\n".join(lines)


def save_dot(path: Path, dot: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dot, encoding="utf-8")