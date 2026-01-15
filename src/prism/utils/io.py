from __future__ import annotations

import csv
import json
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Optional


def _jsonify(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]

    return obj


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(obj), f, indent=2, sort_keys=True)


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
    append: bool = False,
    fieldnames: Optional[list[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    exists = path.exists()
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)