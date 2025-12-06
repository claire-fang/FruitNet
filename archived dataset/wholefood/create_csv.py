
from __future__ import annotations

import csv
from pathlib import Path


def collect_annotations(labels_dir: Path) -> list[list[str]]:
    """Return parsed YOLO annotations from every txt file in labels_dir."""
    rows: list[list[str]] = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        stem = label_file.stem
        with label_file.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"{label_file.name} line {row_index} has {len(parts)} values; expected 5"
                    )
                class_id, x_center, y_center, width, height = parts
                row_idx_str = str(row_index)
                image_path = f"{stem}.jpg"
                label_path = f"{stem}.txt"
                mask_path = f"{stem}*{row_idx_str}*{class_id}.png"
                warm_color_binary = 1
                train_test_validation = 0
                rows.append(
                    [
                        class_id,
                        x_center,
                        y_center,
                        width,
                        height,
                        stem,
                        row_idx_str,
                        image_path,
                        label_path,
                        mask_path,
                        warm_color_binary,
                        train_test_validation,
                    ]
                )
    return rows


def main() -> None:
    labels_dir = Path("labels")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    annotations = collect_annotations(labels_dir)

    output_path = Path("annotations.csv")
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "class_id",
                "x_center",
                "y_center",
                "width",
                "height",
                "file_name",
                "row_index",
                "image_path",
                "label_path",
                "mask_path",
                "warm_color_binary",
                "train_test_validation",
            ]
        )
        writer.writerows(annotations)

    print(f"Wrote {len(annotations)} rows to {output_path}")


if __name__ == "__main__":
    main()
