import os
import shutil
import random
from pathlib import Path

def split_data(dataset_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, output_dir="./split_dataset"):
    """
    Split images and YOLO labels into train/val/test sets.
    
    Directory structure:
    split_dataset/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
    
    Args:
        dataset_root: Path to dataset124 folder
        train_ratio: Proportion of training data (default 0.7)
        val_ratio: Proportion of validation data (default 0.15)
        test_ratio: Proportion of test data (default 0.15)
        seed: Random seed for reproducibility
    """
    
    # Set random seed
    random.seed(seed)
    
    # Verify ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Train, val, and test ratios must sum to 1.0"
    
    # Define paths
    images_dir = Path(dataset_root) / "images"
    # labels_dir = Path(dataset_root) / "binary_yolo"
    # labels_dir = Path(dataset_root) / "yolo_labels"
    labels_dir = Path(dataset_root) / "labels"
    # output_dir = Path(dataset_root) / output_dir
    
    # Create output directories
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted([f for f in images_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices
    total = len(image_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    
    # Copy files to their respective directories
    split_mapping = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    for split, files in split_mapping.items():
        print(f"\nCopying {split} files...")
        for image_file in files:
            # Get corresponding label file
            label_file = labels_dir / (image_file.stem + ".txt")
            
            # Copy image
            dest_image = output_dir / "images" / split / image_file.name
            shutil.copy2(image_file, dest_image)
            
            # Copy label if it exists
            if label_file.exists():
                dest_label = output_dir / "labels" / split / label_file.name
                shutil.copy2(label_file, dest_label)
            else:
                print(f"  Warning: No label found for {image_file.name}")
        
        print(f"  ✓ {split} completed")
    
    print("\n" + "="*50)
    print("Data split completed successfully!")
    print(f"Output directory: {output_dir}")
    print("="*50)
    
    # Print summary
    for split in ["train", "val", "test"]:
        img_count = len(list((output_dir / "images" / split).glob("*")))
        lbl_count = len(list((output_dir / "labels" / split).glob("*")))
        print(f"{split}: {img_count} images, {lbl_count} labels")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test sets")
    parser.add_argument("--dataset_root", type=str, default="./dataset124",
                        help="Path to dataset124 folder")
    parser.add_argument("--train", type=float, default=0.7,
                        help="Training set ratio (default 0.7)")
    parser.add_argument("--val", type=float, default=0.15,
                        help="Validation set ratio (default 0.15)")
    parser.add_argument("--test", type=float, default=0.15,
                        help="Test set ratio (default 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./split_dataset",
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    print(args.dataset_root)
    
    split_data(
        dataset_root=args.dataset_root,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        output_dir=args.output_dir
    )