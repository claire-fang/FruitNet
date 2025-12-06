#!/usr/bin/env python3

"""
Utility for remapping YOLO class ids inside label files.

The default mapping turns the original labels
['38', '-1', '61', '55', '5', '53', '9']
into contiguous ids starting from 0. The labels are remapped in place for the
provided directories (defaults to ./train and ./val).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


DEFAULT_LABELS = ["38", "-1", "61", "55", "5", "53", "9"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remap YOLO class ids in-place for *.txt label files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("train"), Path("val")],
        help="Folders containing YOLO labels (default: %(default)s).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help=(
            "List of original labels ordered by their desired new id. "
            "Example: %(default)s"
        ),
    )
    parser.add_argument(
        "--extension",
        default=".txt",
        help="Only files ending with this suffix will be processed (default: %(default)s).",
    )
    return parser.parse_args()


def remap_file(path: Path, mapping: dict[str, str]) -> int:
    """Apply the mapping to the first column of every line in the file."""
    updated_lines = []
    changed = 0
    with path.open("r", encoding="ascii", errors="ignore") as src:
        for line in src:
            stripped = line.strip()
            if not stripped:
                updated_lines.append(line)
                continue
            parts = stripped.split()
            old_label = parts[0]
            if old_label not in mapping:
                raise ValueError(f"Label '{old_label}' in {path} is not in the mapping.")
            new_label = mapping[old_label]
            if new_label != old_label:
                changed += 1
            parts[0] = new_label
            updated_lines.append(" ".join(parts) + "\n")
    if changed:
        path.write_text("".join(updated_lines), encoding="ascii")
    return changed


def main() -> int:
    args = parse_args()
    mapping = {str(idx): label  for idx, label in enumerate(args.labels)}
    total_files = 0
    total_lines = 0

    for root in args.paths:
        if not root.exists():
            continue
        if root.is_file():
            targets = [root] if root.suffix == args.extension else []
        else:
            targets = sorted(
                p for p in root.rglob(f"*{args.extension}") if p.is_file()
            )
        for label_file in targets:
            total_files += 1
            total_lines += remap_file(label_file, mapping)

    if total_files == 0:
        print("No matching label files were found.", file=sys.stderr)
        return 1
    print(
        f"Remapped labels for {total_lines} entries across {total_files} files.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())