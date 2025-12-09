from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple


def parse_mask_name(filename: str) -> Tuple[str, int, int]:
    """Return (image_id, row_idx, cls_id) from IMAGE*ROW*CLS.png."""
    stem = Path(filename).stem
    image_id, row_idx, cls_id = stem.split("*")
    return image_id, int(row_idx), int(cls_id)


def iter_masks(root: str | Path = "masks_grabcut") -> Iterator[Tuple[str, Path, str, int, int]]:
    """Yield (class_id, mask_path, image_id, row_idx, cls_id) for every mask."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"{root} does not exist")

    for class_dir in sorted(root_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = class_dir.name
        for mask_path in sorted(class_dir.glob("*.png")):
            image_id, row_idx, cls_id = parse_mask_name(mask_path.name)
            yield class_id, mask_path, image_id, row_idx, cls_id


def main() -> None:
    for class_id, mask_path, image_id, row_idx, cls_id in iter_masks():
        print(
            f"class={class_id:>2} | mask={mask_path} | "
            f"image_id={image_id} row={row_idx} cls={cls_id}"
        )


if __name__ == "__main__":
    main()
