from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def regroup_masks(source_root: Path, destination_root: Path, *, copy: bool) -> None:
    """Group mask files by image name and move/copy them into a new folder tree."""
    source_root = source_root.expanduser().resolve()
    destination_root = destination_root.expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source directory {source_root} does not exist")

    destination_root.mkdir(parents=True, exist_ok=True)

    operation = shutil.copy2 if copy else shutil.move
    grouped = 0

    for class_dir in sorted(source_root.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.resolve() == destination_root:
            continue

        for mask_path in class_dir.glob("*.png"):
            image_name = mask_path.name.split("*", 1)[0]
            target_dir = destination_root / image_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / mask_path.name

            if target_path.exists():
                raise FileExistsError(f"Target file {target_path} already exists")

            operation(str(mask_path), str(target_path))
            grouped += 1

    print(f"Grouped {grouped} mask files into {destination_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regroup mask files by image name")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.cwd(),
        help="Root directory that currently contains fruit class folders",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path.cwd() / "grouped_by_image",
        help="Directory where the new grouped folders will be created",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them (default is move)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    regroup_masks(args.source, args.destination, copy=args.copy)
