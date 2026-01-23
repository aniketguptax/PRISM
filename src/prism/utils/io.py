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
        keys = set()
        for r in rows:
            keys.update(r.keys())
        fieldnames = sorted(keys)

    file_exists = path.exists()
    mode = "a" if append else "w"
    
    write_header = (not append) or (not file_exists)
    
    if append and file_exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            
        if existing_header is None:
            write_header = True
        else:
            if list(existing_header) != list(fieldnames):
                raise ValueError(
                    f"CSV header mismatch when appending to {path}.\n"
                    f"Existing: {existing_header}\n"
                    f"New:      {fieldnames}\n"
                    f"Use a fresh outdir or delete the old CSV."
                )
                
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)