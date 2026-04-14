from __future__ import annotations

import sys
from pathlib import Path


def looks_like_data_path(token: str) -> bool:
    text = token.strip()
    if not text or text.startswith("-"):
        return False
    path = Path(text)
    if path.exists() and path.is_file():
        return True
    suffix = path.suffix.lower()
    return suffix in {".csv", ".txt", ".npy"}


def main() -> None:
    argv = sys.argv[1:]
    if argv and looks_like_data_path(argv[0]):
        from prism.auto import main as auto_main

        auto_main()
        return

    from prism.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
