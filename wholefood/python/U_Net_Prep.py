#!/usr/bin/env python3
"""
Generate cropped image / mask patches for U-Net training from YOLO annotations.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageOps


def yolo_box_to_pixels(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Convert normalized YOLO bbox to integer pixel coordinates."""
    x_center_px = x_center * image_width
    y_center_px = y_center * image_height
    half_w = (width * image_width) / 2.0
    half_h = (height * image_height) / 2.0

    x_min = max(0, int(round(x_center_px - half_w)))
    y_min = max(0, int(round(y_center_px - half_h)))
    x_max = min(image_width, int(round(x_center_px + half_w)))
    y_max = min(image_height, int(round(y_center_px + half_h)))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"Invalid crop dimensions computed: ({x_min}, {y_min}, {x_max}, {y_max})"
        )
    return x_min, y_min, x_max, y_max


def ensure_directory(path: Path) -> None:
    """Create directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def guess_mask_folder(row: dict, image_path: Path) -> str:
    """Determine which masks/<folder> to use for this row."""
    for key in ("file_name", "mask_folder", "image_name", "image_id", "mask_path"):
        value = row.get(key)
        if value:
            return Path(value).stem
    return image_path.stem


def read_float(row: dict, key: str) -> float:
    if key not in row:
        raise KeyError(f"'{key}' not found in CSV header")
    return float(row[key])


def collect_mask_files(mask_root: Path, folder_name: str) -> List[Path]:
    candidate_dir = mask_root / folder_name
    if not candidate_dir.exists():
        return []
    files = [p for p in candidate_dir.iterdir() if p.is_file()]
    return sorted(files)


def select_mask_files(
    row: dict, mask_root: Path, mask_folder: str, row_number: int
) -> List[Path]:
    mask_value = row.get("mask_path") or row.get("mask_file") or row.get("mask_filename")
    if mask_value:
        mask_name = Path(mask_value).name
        candidates = [
            mask_root / mask_folder / mask_name,
            mask_root / mask_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return [candidate]
        print(
            f"[WARN] Specified mask '{mask_value}' not found for row {row_number}; skipping."
        )
        return []
    return collect_mask_files(mask_root, mask_folder)


def crop_unet_patches(
    annotations_csv: Path,
    images_dir: Path,
    masks_dir: Path,
    output_images_dir: Path,
    output_masks_dir: Path,
) -> None:
    ensure_directory(output_images_dir)
    ensure_directory(output_masks_dir)

    total_rows = 0
    saved_images = 0
    saved_masks = 0
    skipped_rows: List[str] = []

    with annotations_csv.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=1):
            total_rows += 1
            try:
                image_rel = (
                    row.get("image_path")
                    or row.get("image")
                    or row.get("image_file")
                    or row.get("image_filename")
                )
                if not image_rel:
                    raise ValueError("Missing image_path column value")
                image_path = images_dir / image_rel
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                image = Image.open(image_path).convert("RGB")
                image = ImageOps.exif_transpose(image)
                bbox = yolo_box_to_pixels(
                    read_float(row, "x_center"),
                    read_float(row, "y_center"),
                    read_float(row, "width"),
                    read_float(row, "height"),
                    image.width,
                    image.height,
                )

                mask_folder = guess_mask_folder(row, image_path)
                mask_files = select_mask_files(row, masks_dir, mask_folder, row_number)
                if not mask_files:
                    print(f"[WARN] No masks for {mask_folder} (row {row_number})")
                    continue

                crop = image.crop(bbox)

                for mask_file in mask_files:
                    output_stem = mask_file.stem
                    image_output_path = output_images_dir / f"{output_stem}.png"
                    crop.save(image_output_path)
                    saved_images += 1

                    mask_image = Image.open(mask_file)
                    mask_image = ImageOps.exif_transpose(mask_image)
                    if mask_image.size != image.size and mask_image.size[::-1] == image.size:
                        mask_image = mask_image.transpose(Image.Transpose.ROTATE_90)
                        if mask_image.size != image.size:
                            mask_image = mask_image.transpose(Image.Transpose.ROTATE_180)
                    if mask_image.size != image.size:
                        print(
                            f"[WARN] Mask {mask_file} size {mask_image.size} "
                            f"differs from image {image.size}; skipping."
                        )
                        continue
                    mask_crop = mask_image.crop(bbox)
                    mask_output_path = output_masks_dir / f"{output_stem}.png"
                    mask_crop.save(mask_output_path)
                    saved_masks += 1

            except Exception as exc:  # noqa: BLE001 - log and continue
                skipped_rows.append(f"row {row_number}: {exc}")
                print(f"[ERROR] Skipping row {row_number}: {exc}")

    print(
        "Finished processing.\n"
        f"  Rows read: {total_rows}\n"
        f"  Image crops saved: {saved_images}\n"
        f"  Mask crops saved: {saved_masks}\n"
        f"  Rows skipped: {len(skipped_rows)}"
    )
    if skipped_rows:
        print("Skipped details:")
        for msg in skipped_rows:
            print(f"    - {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cropped image/mask patches for U-Net training."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("wholefood_annotations.csv"),
        help="CSV file with YOLO annotations.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images"),
        help="Directory containing the source RGB images.",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=Path("masks"),
        help="Directory containing sub-folders of masks per image.",
    )
    parser.add_argument(
        "--output-images",
        type=Path,
        default=Path("unet_crops/images"),
        help="Destination for cropped RGB patches.",
    )
    parser.add_argument(
        "--output-masks",
        type=Path,
        default=Path("unet_crops/masks"),
        help="Destination for cropped mask patches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crop_unet_patches(
        annotations_csv=args.annotations,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_images_dir=args.output_images,
        output_masks_dir=args.output_masks,
    )


if __name__ == "__main__":
    main()
