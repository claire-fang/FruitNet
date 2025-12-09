#!/usr/bin/env python3

"""
Rename image files so they only keep the trailing IMG_#### portion.

Example:
    `1103042_dataset 2025-12-04 20-38-37_IMG_3362.jpg`
becomes
    `IMG_3362.jpg`
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

IMG_PATTERN = re.compile(r"(IMG_\d+)")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename files by keeping only the trailing IMG_#### portion."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory containing the files to rename (default: %(default)s).",
    )
    parser.add_argument(
        "--suffix",
        default=".jpg",
        help="Only rename files with this suffix (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned renames without modifying the filesystem.",
    )
    return parser.parse_args()


def iter_targets(root: Path, suffix: str) -> list[Path]:
    """Yield files under root that end with suffix."""
    if root.is_file():
        return [root] if root.suffix == suffix else []
    return sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        raise SystemExit(f"{args.root} does not exist. Pass a path or run with --root.")

    for path in iter_targets(args.root, args.suffix):
        match = IMG_PATTERN.search(path.name)
        if not match:
            raise SystemExit(f"Could not find IMG_#### pattern in '{path.name}'.")
        target = path.with_name(f"{match.group(1)}{path.suffix}")
        if target.exists() and target != path:
            raise SystemExit(f"Cannot rename '{path.name}' -> '{target.name}': target exists.")
        if args.dry_run:
            print(f"{path.name} -> {target.name}")
        else:
            path.rename(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
