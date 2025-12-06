# FruitNet - YOLO/UNet-based Fruit Detection

A comprehensive fruit detection system using YOLOv8 for real-time object detection in agricultural settings.

## Project Overview

FruitNet is designed to detect and localize fruits in images using the YOLOv8 object detection architecture. The project includes data preprocessing, dataset organization, model training, and evaluation capabilities.

## Project Structure

```
FruitNet/
├── dataset124/
│   ├── images/                 # Original images
│   ├── binary_yolo/            # YOLO annotation labels
│   ├── split_dataset/          # Organized train/val/test split
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── data.yaml           # YOLO dataset configuration
│   ├── yolo_labels/            # Additional label files
│   └── masks/                  # Segmentation masks
├── Preprocessing Steps/
│   ├── dataprep.py             # Data preparation
│   ├── process_masks.py        # Mask processing
│   ├── remap.py                # Label remapping
│   ├── rename.py               # File renaming
│   └── sort.py                 # File sorting
├── train.py                    # Main training/validation script
├── split_data.py               # Dataset splitting utility
├── CS230_Final_Project.ipynb   # Project notebook
└── README.md                   # This file
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Virtual environment (recommended)

### Virtual Environment Setup

```bash
python3 -m venv fnvenv
source fnvenv/bin/activate
```

### Install Dependencies
Our YOLO model was trained on NVIDIA B200 GPUs. The dependency versions required for the B200 are listed below. If you are using a different GPU, please refer to the appropriate versions for your setup.

```bash
pip install numpy==1.26.4
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install ultralytics
```

## Data Organization

### Preparing Your Dataset

Use the `split_data.py` script to organize your data into train/val/test splits:

```bash
python split_data.py --dataset_root "PATH TO YOUR DATA" \
                     --train 0.7 \
                     --val 0.15 \
                     --test 0.15 \
                     --seed 42
```

**Arguments:**
- `--dataset_root`: Path to your dataset folder
- `--train`: Training set ratio (default: 0.7)
- `--val`: Validation set ratio (default: 0.15)
- `--test`: Test set ratio (default: 0.15)
- `--seed`: Random seed for reproducibility (default: 42)

This creates:
```
split_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## Training & Validation

### Dataset Configuration

The `data.yaml` file in `split_dataset/` contains dataset paths and class information:

```yaml
path: /data/fjx/FruitNet/dataset124/split_dataset
train: images/train
val: images/val
test: images/test
nc: 1
names: ['fruit']
```

### Training

Start training with:

```bash
python train.py --mode train --epochs 50 --imgsz 640
```

**Training Arguments:**
- `--mode train`: Enable training mode (required)
- `--epochs`: Number of training epochs (default: 50)
- `--imgsz`: Input image size (default: 640)
- `--model`: Model to use (default: yolov8n.pt)
- `--data`: Path to data.yaml (default: ./split_dataset/data.yaml)

### Validation/Testing

Evaluate your model on the test set:

```bash
python train.py --mode val
```

**Validation Arguments:**
- `--mode val`: Enable validation mode (required)
- `--weights`: Path to trained weights (default: runs/detect/train7/weights/best.pt)
- `--imgsz`: Input image size (default: 640)
- `--data`: Path to data.yaml (default: ./split_dataset/data.yaml)

### Custom Model Weights

Use different model weights for validation:

```bash
python train.py --mode val --weights path/to/your/weights.pt
```

## GPU Information

Check GPU availability and status:

```bash
nvidia-smi
```

The training script automatically detects and uses CUDA if available.

## Output

- **Training outputs**: Saved in `runs/detect/train*/` directories
- **Best weights**: Available at `runs/detect/train*/weights/best.pt`
- **Validation results**: Printed to console and saved in results files

## Data Preprocessing

Located in `Preprocessing Steps/`, several utility scripts are available:

- `dataprep.py`: Prepare raw data
- `process_masks.py`: Process segmentation masks
- `remap.py`: Remap labels
- `rename.py`: Rename files systematically
- `sort.py`: Sort data files

Run preprocessing scripts as needed:

```bash
python Preprocessing\ Steps/dataprep.py
```

## Supported Models

- **yolov8n.pt**: Nano (fastest, least accurate)
- **yolov8s.pt**: Small
- **yolov8m.pt**: Medium
- **yolov8l.pt**: Large
- **yolov8x.pt**: Extra Large (slowest, most accurate)

Specify model size during training:

```bash
python train.py --mode train --model yolov8m.pt --epochs 100
```

## Performance Metrics

The model outputs standard YOLO metrics:
- **mAP50**: Mean Average Precision at 0.5 IoU
- **mAP50-95**: Mean Average Precision at 0.5-0.95 IoU
- **Precision & Recall**: Per-class metrics

## Troubleshooting

### Import Errors
Ensure all packages are installed:
```bash
pip install ultralytics torch opencv-python
```

### GPU Not Available
Check CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### File Not Found
Ensure you're running from the correct directory:
```bash
cd /data/fjx/FruitNet
```

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

## License

This project is for educational purposes. Refer to the CS230 Final Project specifications for license details.

## Contact & Support

For questions or issues, refer to the CS230_Final_Project.ipynb notebook for detailed implementation notes.

