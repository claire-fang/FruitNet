#!/usr/bin/env python3
"""
Rearrange JPG and TXT files into train/val/test folders based on source_annotations.csv.
"""

import csv
import shutil
from collections import defaultdict
from pathlib import Path


def ensure_split_dirs(base: Path, splits: dict) -> None:
    """Ensure train/val/test directories exist for both images and labels."""
    images_base = base / "images"
    labels_base = base / "labels"
    for split_name in splits.values():
        (images_base / split_name).mkdir(parents=True, exist_ok=True)
        (labels_base / split_name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    csv_path = repo_root / "source_annotations.csv"
    images_dir = repo_root / "images"
    labels_dir = repo_root / "labels"
    output_root = repo_root / "train_test_val"

    split_map = {0: "train", 1: "val", 2: "test"}
    ensure_split_dirs(output_root, split_map)

    processed_pairs = set()
    assigned_split = {}
    copy_counts = defaultdict(int)
    warnings = []

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file at {csv_path}")

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"train_test_validation", "image_path", "label_path"}
        if not required_columns.issubset(reader.fieldnames or []):
            missing_cols = required_columns - set(reader.fieldnames or [])
            raise ValueError(f"Missing expected columns in CSV: {missing_cols}")

        for row_index, row in enumerate(reader, start=1):
            split_value = row.get("train_test_validation")
            image_name = (row.get("image_path") or "").strip()
            label_name = (row.get("label_path") or "").strip()

            try:
                split_key = int(split_value)
            except (TypeError, ValueError):
                warnings.append(
                    f"Row {row_index}: invalid train_test_validation value '{split_value}'"
                )
                continue

            if split_key not in split_map:
                warnings.append(
                    f"Row {row_index}: unexpected split value '{split_key}' for {image_name}"
                )
                continue

            if not image_name or not label_name:
                warnings.append(f"Row {row_index}: missing image or label file name")
                continue

            split_name = split_map[split_key]
            assignment = assigned_split.setdefault(image_name, split_name)
            if assignment != split_name:
                warnings.append(
                    f"Row {row_index}: image {image_name} previously assigned to '{assignment}'"
                )
                continue

            pair_key = (image_name, label_name, split_name)
            if pair_key in processed_pairs:
                continue  # already copied this pair for the split

            image_src = images_dir / image_name
            label_src = labels_dir / label_name
            if not image_src.exists():
                warnings.append(f"Row {row_index}: missing image file {image_src}")
                continue
            if not label_src.exists():
                warnings.append(f"Row {row_index}: missing label file {label_src}")
                continue

            image_dst = output_root / "images" / split_name / image_name
            label_dst = output_root / "labels" / split_name / label_name

            shutil.copy2(image_src, image_dst)
            shutil.copy2(label_src, label_dst)

            processed_pairs.add(pair_key)
            copy_counts[split_name] += 1

    total_copied = sum(copy_counts.values())
    print(f"Copied {total_copied} image/label pairs:")
    for split_name in split_map.values():
        print(f"  {split_name}: {copy_counts[split_name]} pairs")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
