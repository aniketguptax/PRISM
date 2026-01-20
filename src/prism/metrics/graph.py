from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Iterable, Optional

from prism.reconstruction.protocols import PredictiveStateModel

# (state, symbol, next_state, probability)
Edge = Tuple[int, int, int, float]  


def to_edge_list(model: PredictiveStateModel) -> List[Edge]:
    edges: List[Edge] = []
    for (s, sym), sp_dict in model.transitions.items():
        for sp, prob in sp_dict.items():
            edges.append((s, sym, sp, float(prob)))
    return edges


def to_dot(
    edges: Iterable[Edge],
    graph_name: str = "state_machine",
    rankdir: str = "LR",
    label: Optional[str] = None,
    prob_precision: int = 3,
) -> str:
    edges = list(edges)
    nodes = sorted({s for s, _, _, _ in edges} | {sp for _, _, sp, _ in edges})

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
        
    def format_prob(p: float) -> str:
        return f"{p:.{prob_precision}}"

    for s, sym, sp, p in sorted(edges, key=lambda e: (e[0], e[1], e[2])):
        lines.append(f'  {s} -> {sp} [label="{sym}: {format_prob(p)}"];')

    lines.append("}")
    return "\n".join(lines)


def save_dot(path: Path, dot: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dot, encoding="utf-8")