# FruitNet - Multi-Model Fruit Detection System

A comprehensive fruit detection and segmentation system supporting both **YOLOv8 (object detection)** and **U-Net (semantic segmentation)** for agricultural applications. This project includes complete data preprocessing, dataset organization, multi-modal training, and evaluation pipelines.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Data Preparation](#data-preparation--management)
- [Training Models](#training-models)
- [GPU & Hardware](#gpu--hardware-information)
- [Output & Results](#training-output--results)
- [Advanced Usage](#advanced-data-preprocessing)
- [Troubleshooting](#troubleshooting)

## Project Overview

FruitNet provides a complete end-to-end solution for fruit detection in images:

- **YOLOv8**: Real-time bounding box detection with class labels
- **U-Net**: Precise pixel-level semantic segmentation for individual fruit identification
- **Data Management**: Automated splitting, annotation tracking, and preprocessing
- **Training Pipelines**: Flexible training/validation modes with GPU acceleration
- **Annotation Tools**: CSV-based tracking with YOLO labels, warm color detection, and mask data

## Project Structure

### Directory Tree

```
FruitNet/
├── Core Training Scripts
│   ├── yolo.py                       # Alternative YOLO training script
│   └── unet.py                       # U-Net segmentation model
│
├── Data Processing & Management
│   ├── split_data.py                 # Dataset splitting (train/val/test)
│   ├── update_annotations_split.py   # Update train/val/test assignments in CSV
│   └── source_annotations.csv        # Master annotation file (11,850 rows)
│
├── Dataset Directory (dataset124/)
│   ├── images/                       # 2000 fruit images (JPG)
│   ├── binary_yolo/                  # YOLO format labels (class_id, x, y, w, h)
│   ├── yolo_labels/                  # Alternative YOLO label format with class info
│   ├── masks/                        # Segmentation masks grouped by image
│   │   ├── 0/                    # Image folder (named by image ID)
│   │   │   ├── 0*1*5.png         # Mask format: image*instance*class_id
│   │   │   ├── 0*2*5.png
│   │   │   └── ...
│   │   └── ...
│   │
│   └── change.py                     # Dataset modification utility
│
├── split_dataset/                    # Organized train/val/test structure for YOLO
│   ├── images/
│   │   ├── train/                    # ~1400 training images
│   │   ├── val/                      # ~300 validation images
│   │   └── test/                     # ~300 test images
│   ├── labels/
│   │   ├── train/                    # Corresponding YOLO labels
│   │   ├── val/
│   │   └── test/
│   └── dataset.yaml                  # YOLO data config
│
├── Preprocessing Scripts (Preprocessing Steps/)
│   ├── create_csv.py                 # Create centralized dataset for faster processing 
│   ├── dataprep.py                   # Raw data preparation
│   ├── process_masks.py              # Segmentation mask processing
│   ├── remap.py                      # Label class remapping
│   ├── rename.py                     # Batch file renaming
│   └── sort.py                       # Data organization/sorting
│
├── Training Outputs
│   ├── runs/                         # Model training results
│   │   ├── detect/
│   │   │   ├── train*/               # Individual training run folders
│   │   │   │   ├── weights/
│   │   │   │   │   ├── best.pt       # Best model weights
│   │   │   │   │   └── last.pt       # Latest checkpoint
│   │   │   │   ├── results.csv
│   │   │   │   └── metrics/
│   │   │   └── test/                 # Test results
│   │   └── ...
│
└── Documentation
    └── README.md                     # This file
```

### Key Directories Explained

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `dataset124/images/` | 2000 JPG images | Raw input images |
| `dataset124/binary_yolo/` | YOLO TXT labels | Bounding box annotations (binary format) |
| `dataset124/yolo_labels/` | YOLO TXT labels | Class-aware bounding boxes |
| `dataset124/masks/` | PNG masks per image | Pixel-level segmentation masks |
| `split_dataset/images/` | Organized images | Train/val/test split (70/15/15) |
| `split_dataset/labels/` | Organized labels | Corresponding YOLO labels for each split |
| `runs/detect/train*/` | Model weights & logs | Training results and best models |

## Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ or CUDA 12.8 (for GPU acceleration)
- Virtual environment (recommended)
- NVIDIA GPU (tested on B200; other GPUs supported)

### Virtual Environment Setup

**Create new environment**
```bash
python3 -m venv fnvenv
source fnvenv/bin/activate  # On Linux/Mac
# or
fnvenv\Scripts\activate     # On Windows
```

### Install Dependencies

For **NVIDIA B200 GPUs** (PyTorch nightly with CUDA 12.8):
```bash
pip install numpy==1.26.4
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install ultralytics
pip install tensorflow  # For U-Net model (optional)
pip install pandas opencv-python matplotlib
```
If you are using a different GPU, please refer to the appropriate versions for your setup.

## Data Preparation & Management

### 1. Splitting Dataset into Train/Val/Test

Use `split_data.py` to organize images and labels into splits:

```bash
python split_data.py --dataset_root /path/to/dataset \
                     --train 0.7 \
                     --val 0.15 \
                     --test 0.15 \
                     --seed 42
```

**Arguments:**
- `--dataset_root`: Path to dataset124 folder (default: `./dataset124`)
- `--train`: Training set ratio (default: 0.7 = 70%)
- `--val`: Validation set ratio (default: 0.15 = 15%)
- `--test`: Test set ratio (default: 0.15 = 15%)
- `--seed`: Random seed for reproducibility (default: 42)

**Output Structure:**
```
split_dataset/
├── images/
│   ├── train/ 
│   ├── val/ 
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 2. Annotation Management

#### Build CSV from Split Dataset Labels (Preprocessing Steps/)

```bash
python create_csv.py
```

**CSV columns:**
- `class_id`: Fruit class (from yolo_labels/)
- `x_center, y_center, width, height`: Normalized YOLO coordinates
- `file_name`: Image identifier
- `image_path, label_path, mask_path`: File paths
- `warm_color_binary`: Whether warm colors detected (1=yes, 0=no)
- `train_test_validation`: Split assignment (0=train, 1=val, 2=test)

#### Update Train/Val/Test Assignments

Synchronize annotations with actual split folders:

```bash
python update_annotations_split.py --source_csv source_annotations.csv \
                                   --split_labels split_dataset/labels \
                                   --output updated_annotations.csv
```

This updates the `train_test_validation` column (0, 1, or 2) based on which split folder the label file is in.

### 3. Data Preprocessing (Optional)

Located in `Preprocessing Steps/`, use these utilities for custom preprocessing:

```bash
# Prepare raw data
python Preprocessing\ Steps/dataprep.py

# Process segmentation masks
python Preprocessing\ Steps/process_masks.py

# Remap class labels
python Preprocessing\ Steps/remap.py

# Batch rename files
python Preprocessing\ Steps/rename.py

# Sort and organize data
python Preprocessing\ Steps/sort.py
```

## Training Models

### YOLOv8 Object Detection Training

Train the YOLO model on your dataset:

```bash
python train.py --mode train
```

**Training Process:**
1. Loads configuration from `split_dataset/dataset.yaml`
2. Trains on images in `split_dataset/images/train/`
3. Validates on images in `split_dataset/images/val/`
4. Saves best model checkpoint to `runs/detect/train<N>/weights/best.pt`
5. Generates training metrics and visualizations

**Default Parameters:**
- Epochs: 50
- Image size: 640
- Weights: `runs/detect/train20/weights/best.pt` (pre-trained)
- Batch size: 16 (determined by available GPU memory)
- Device: auto (GPU if available, CPU otherwise)

**Custom Training Arguments:**
```bash
python train.py --mode train --epochs 100 --imgsz 832 --batch 16
```

### YOLOv8 Validation

Run validation on the test set:

```bash
python train.py --mode test
```

**Validation Outputs:**
- Precision, Recall, mAP50, mAP50-95 metrics
- Confusion matrices
- Validation images with predictions
- Results saved to `runs/detect/train<N>/`

### U-Net Semantic Segmentation

For pixel-level fruit detection, use the U-Net model:

```bash
python unet.py
```

**U-Net Architecture:**
- Input: 96×128 RGB images
- Output: 67-class semantic segmentation
- Encoder-decoder with skip connections
- Training data: Cropped regions from YOLO boxes + mask annotations

**Training Process (from unet.py):**
1. Load images and YOLO bounding boxes
2. Crop regions and mask annotations
3. Preprocess to 96×128 resolution
4. Train encoder-decoder with categorical crossentropy
5. Validate on holdout set

**Metrics to Monitor:**
- Loss (training and validation)
- Accuracy
- Precision and Recall
- mAP (mean Average Precision)
- GPU memory usage
- Training speed (images/sec)

## License

This project is for educational purposes. Refer to the CS230 Final Project specifications for license details.