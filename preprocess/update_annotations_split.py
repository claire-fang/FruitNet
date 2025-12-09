import csv
import os
from pathlib import Path

def update_annotations_split(
    source_csv_path,
    split_labels_dir,
    output_csv_path=None
):
    """
    Update source_annotations.csv with correct train/test/validation assignments
    based on which split the label file is in (split_dataset/labels/train/val/test).
    
    Args:
        source_csv_path: Path to source_annotations.csv
        split_labels_dir: Path to split_dataset/labels directory
        output_csv_path: Output CSV path (default: overwrites source)
    """
    
    if output_csv_path is None:
        output_csv_path = source_csv_path
    
    split_labels_dir = Path(split_labels_dir)
    
    # Map each image to its split
    image_to_split = {}
    
    # Map split folder name to value: 0=train, 1=val, 2=test
    split_map = {"train": 0, "val": 1, "test": 2}
    
    print("Scanning split_dataset/labels for image assignments...")
    
    for split_type, split_value in split_map.items():
        split_folder = split_labels_dir / split_type
        if split_folder.exists():
            label_files = list(split_folder.glob("*.txt"))
            for label_file in label_files:
                image_name = label_file.stem
                image_to_split[image_name] = split_value
            print(f"  Found {len(label_files)} files in {split_type}/")
    
    print(f"\nTotal images found: {len(image_to_split)}")
    # print(image_to_split.keys())
    
    # Read source CSV
    print(f"\nReading source annotations from: {source_csv_path}")
    rows = []
    with open(source_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Total rows in source CSV: {len(rows)}")
    
    # Update train_test_validation column
    updated_count = 0
    no_match_count = 0
    
    for row in rows:
        file_name = row.get('file_name')
        # print(file_name)
        
        if file_name in image_to_split:
            old_value = row.get('train_test_validation', '')
            new_value = image_to_split[file_name]
            row['train_test_validation'] = str(new_value)
            
            if old_value != str(new_value):
                updated_count += 1
        else:
            no_match_count += 1
    
    print(f"\nUpdates:")
    print(f"  Rows updated: {updated_count}")
    print(f"  Rows with no match: {no_match_count}")
    
    # Write updated CSV
    if rows:
        fieldnames = list(rows[0].keys())
        
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✓ Updated CSV written to: {output_csv_path}")
        
        # Print summary statistics
        train_count = sum(1 for r in rows if r['train_test_validation'] == '0')
        val_count = sum(1 for r in rows if r['train_test_validation'] == '1')
        test_count = sum(1 for r in rows if r['train_test_validation'] == '2')
        
        print(f"\nSplit Distribution:")
        print(f"  Train (0): {train_count}")
        print(f"  Val (1): {val_count}")
        print(f"  Test (2): {test_count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Update source_annotations.csv with correct train/val/test split assignments"
    )
    parser.add_argument(
        "--source_csv",
        type=str,
        default="./source_annotations.csv",
        help="Path to source_annotations.csv"
    )
    parser.add_argument(
        "--split_labels",
        type=str,
        default="./split_dataset/labels",
        help="Path to split_dataset/labels directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: overwrites source)"
    )
    
    args = parser.parse_args()
    
    update_annotations_split(
        args.source_csv,
        args.split_labels,
        args.output
    )
