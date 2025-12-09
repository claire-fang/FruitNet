from pathlib import Path


POSITIVE_CLASS_IDS = {
    2, 5, 6, 9, 14, 25, 37, 38, 39, 44, 45, 48, 49, 53, 55, 57, 61, 62, 63, 64, 66
}


def remap_class_id(value: str) -> str:
    """Return the mapped class id as a string, defaulting to 0."""
    try:
        class_id = int(value)
    except ValueError:
        raise ValueError(f"Invalid class id '{value}'") from None
    return "1" if class_id in POSITIVE_CLASS_IDS else "0"


def process_file(path: Path) -> None:
    new_lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split()
        parts[0] = remap_class_id(parts[0])
        new_lines.append(" ".join(parts))
    path.write_text("\n".join(new_lines) + "\n")


def main() -> None:
    labels_dir = Path(__file__).resolve().parent / "binary_yolo"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {labels_dir}")
    for txt_file in labels_dir.glob("*.txt"):
        process_file(txt_file)


if __name__ == "__main__":
    main()
