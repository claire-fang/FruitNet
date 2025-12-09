import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from unet_torch import FruitNetDataset, pixel_accuracy



def test_yolo(model_path, test_data_path, output_csv_path = "yolo_test_predictions.csv"):
    model = YOLO(model_path)

    all_boxes = []

    for result in model.predict(source=test_data_path, imgsz=640, stream=True, conf=0.001, iou=0.6, device=4):
        # format: x_c, y_c, w, h
        xywh = result.boxes.xywh.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()
        file = Path(result.path)
        
        for i in range(len(xywh)):
            file_name = file.stem
            image_path = f"{file.stem}.jpg"
            label_path = f"{file.stem}.txt"
            if cls[i] == 1.0:
                mask_path = f"{file.stem}.png"
            else:
                mask_path = None
            all_boxes.append([
                xywh[i][0], xywh[i][1], xywh[i][2], xywh[i][3], # Coordinates
                file_name, # Image filename
                i, # row index
                image_path,
                label_path,
                mask_path,
                int(cls[i]), # warm_color_binary,
                2 # train_test_validation (test set)
            ])
    columns = ["x_center", "y_center", "width", "height", "file_name", "row_index", "image_path", 
               "label_path", "mask_path","warm_color_binary", "train_test_validation"]
    df = pd.DataFrame(all_boxes, columns=columns)
    print(df.head())
    df.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to {output_csv_path}")


def test_model(model_path, root_path="./dataset124"):
    print(f"Loading checkpoint: {model_path}")

    # Dataset
    test_ds = FruitNetDataset(root_path=root_path, mode="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Model
    model = UNet(n_classes=67).cuda()
    model = torch.nn.DataParallel(model)

    # Load checkpoint (supports DataParallel and normal)
    state_dict = torch.load(model_path)
    model.module.load_state_dict(state_dict)

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.cuda()
            masks = masks.squeeze(1).long().cuda()

            preds = model(images)

            loss = criterion(preds, masks)
            acc = pixel_accuracy(preds, masks, ignore_index=255)

            total_loss += loss.item()
            total_acc += acc

    mean_loss = total_loss / len(test_loader)
    mean_acc = total_acc / len(test_loader)

    print("\n=== Test Results ===")
    print(f"Loss: {mean_loss:.4f}")
    print(f"Pixel Accuracy: {mean_acc:.4f}")


if __name__ == "__main__":
    # Example: test last epoch
    # test_model("unet_checkpoints/unet_epoch_9.pth")
    test_yolo(
        model_path="runs/detect/train20/weights/best.pt",
        test_data_path="./split_dataset/images/test"
    )
