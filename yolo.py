import torch
import argparse
from ultralytics import YOLO

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print(torch.version.cuda)
print("Compute capability:", torch.cuda.get_device_capability())

def main():
    parser = argparse.ArgumentParser(description="Train or test YOLO model")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Mode: 'train' for training or 'test' for testing")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--model", type=str, default="yolov8n_fruit_p2.yaml",
                        help="Model to use for training (yolov8n_fruit_p2.yaml)")
    parser.add_argument("--weights", type=str, default="runs/detect/train20/weights/best.pt",
                        help="Weights to use for validation (default: runs/detect/train20/weights/best.pt)")
    parser.add_argument("--data", type=str, default="./split_dataset/dataset.yaml",
                        help="Path to data.yaml (default: ./split_dataset/dataset.yaml)")
    parser.add_argument("--test_data", type=str, default="./wholefood/dataset.yaml",
                        help="Path to data.yaml (default: ./wholefood/dataset.yaml")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("\n" + "="*50)
        print("Starting TRAINING mode")
        print("="*50)
        model = YOLO(args.model)
        
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=16, 
            # --- SELECT GPU HERE ---
            device=[4, 5],   # Use only GPU 4 and 5
        )
        print("\nTraining completed!")
        print(results)
    
    elif args.mode == "test":
        print("\n" + "="*50)
        print("Starting TEST mode")
        print("="*50)
        model = YOLO(args.weights)
        
        # Evaluate on the test set
        # results = model.val(
        #     data=args.data,
        #     task="test",
        #     imgsz=args.imgsz
        # )
        results = model.val(
            data=args.test_data,  # Your Data YAML
            split="test",          # Use the 'test' set (not validation)
            imgsz=640,
            batch=16,
            conf=0.001,            # Validation uses very low conf to capture all candidates
            iou=0.6,
            device=4               # Use your free GPU
        )
        print("\nTest completed!")
        print(results)

if __name__ == "__main__":
    main()