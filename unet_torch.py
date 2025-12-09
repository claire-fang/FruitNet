import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

def convert_yolo_box_to_corners(xc, yc, bw, bh, W, H):
    xc = xc * W
    yc = yc * H
    bw = bw * W
    bh = bh * H

    x1 = (xc - bw/2).long().clamp(0, W-1)
    y1 = (yc - bh/2).long().clamp(0, H-1)
    x2 = (xc + bw/2).long().clamp(0, W-1)
    y2 = (yc + bh/2).long().clamp(0, H-1)

    return x1, y1, x2, y2

def pixel_accuracy(preds, masks, ignore_index=255):
    """
    preds: B×C×H×W logits
    masks: B×1×H×W ground truth
    """
    # Get predicted class per pixel
    preds_class = preds.argmax(dim=1)  # B×H×W
    masks = masks.squeeze(1)           # B×H×W

    # Create mask of valid pixels
    valid = (masks != ignore_index)

    correct = (preds_class == masks) & valid
    acc = correct.sum().float() / valid.sum().float()
    return acc.item()


class FruitNetDataset(Dataset):
    def __init__(self, root_path, mode="train", transform=None, n_classes=67, ignore_index=255):
        self.root = root_path
        self.transform = transform
        self.n_classes = n_classes
        self.ignore_index = ignore_index

        # Load CSV
        df = pd.read_csv(os.path.join(root_path, "source_annotations.csv"))
        df = df[(df["warm_color_binary"] == 1) & (df["mask_path"].notna())]

        if mode == "train":
            df = df[df["train_test_validation"] == 0]
        elif mode == "dev":
            df = df[df["train_test_validation"] == 1]
        else:
            df = df[df["train_test_validation"] == 2]

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return np.array(img)

    def load_mask(self, path):
        mask = Image.open(path).convert("RGB")
        mask = np.array(mask)
        mask = np.max(mask, axis=-1)  # reduce 3 channels → 1

        # Handle out-of-range values
        mask = np.where((mask >= 0) & (mask < self.n_classes), mask, self.ignore_index)

        return mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = f"{self.root}/images/{row['image_path']}"
        mask_path = f"{self.root}/masks_agg/{row['file_name']}.png"

        img = self.load_image(img_path)
        mask = self.load_mask(mask_path)

        H, W = img.shape[:2]

        # YOLO bounding box crop
        xc, yc, bw, bh = row["x_center"], row["y_center"], row["width"], row["height"]
        x1, y1, x2, y2 = convert_yolo_box_to_corners(
            torch.tensor(xc), torch.tensor(yc),
            torch.tensor(bw), torch.tensor(bh),
            W, H
        )

        img_crop = img[y1:y2+1, x1:x2+1]
        mask_crop = mask[y1:y2+1, x1:x2+1]

        # Resize to 96×128
        img_crop = Image.fromarray((img_crop).astype(np.uint8)).resize((128, 96), Image.NEAREST)
        mask_crop = Image.fromarray(mask_crop.astype(np.uint8)).resize((128, 96), Image.NEAREST)

        img_crop = np.array(img_crop) / 255.0
        mask_crop = np.array(mask_crop)

        img_crop = torch.tensor(img_crop).permute(2, 0, 1).float()  # HWC → CHW
        mask_crop = torch.tensor(mask_crop).long().unsqueeze(0)      # 1×H×W

        return img_crop, mask_crop


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_classes=23, n_filters=32):
        super().__init__()

        self.down1 = DoubleConv(3, n_filters)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(n_filters, n_filters*2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(n_filters*2, n_filters*4)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(n_filters*4, n_filters*8, dropout=0.3)
        self.pool4 = nn.MaxPool2d(2)

        self.bottom = DoubleConv(n_filters*8, n_filters*16, dropout=0.3)

        self.up6 = nn.ConvTranspose2d(n_filters*16, n_filters*8, 2, stride=2)
        self.conv6 = DoubleConv(n_filters*16, n_filters*8)

        self.up7 = nn.ConvTranspose2d(n_filters*8, n_filters*4, 2, stride=2)
        self.conv7 = DoubleConv(n_filters*8, n_filters*4)

        self.up8 = nn.ConvTranspose2d(n_filters*4, n_filters*2, 2, stride=2)
        self.conv8 = DoubleConv(n_filters*4, n_filters*2)

        self.up9 = nn.ConvTranspose2d(n_filters*2, n_filters, 2, stride=2)
        self.conv9 = DoubleConv(n_filters*2, n_filters)

        self.out_conv = nn.Conv2d(n_filters, n_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        bottom = self.bottom(p4)

        u6 = self.up6(bottom)
        u6 = torch.cat([u6, c4], dim=1)
        u6 = self.conv6(u6)

        u7 = self.up7(u6)
        u7 = torch.cat([u7, c3], dim=1)
        u7 = self.conv7(u7)

        u8 = self.up8(u7)
        u8 = torch.cat([u8, c2], dim=1)
        u8 = self.conv8(u8)

        u9 = self.up9(u8)
        u9 = torch.cat([u9, c1], dim=1)
        u9 = self.conv9(u9)

        return self.out_conv(u9)

def main():
    train_ds = FruitNetDataset(root_path="./dataset124", mode="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    model = UNet(n_classes=67).cuda()
    model = torch.nn.DataParallel(model)
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=255) # ignore the background

    for epoch in range(100):
        epoch_acc = 0.0
        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.squeeze(1).long().cuda()  # shape B×H×W

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        save_path = f"unet_checkpoints/unet_epoch_{epoch}.pth"
        torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) 
                else model.state_dict(), 
                save_path)
        print(f"Saved model to {save_path}")

        epoch_acc += pixel_accuracy(preds, masks)
        print(f"Epoch {epoch} pixel accuracy = {epoch_acc:.4f}")
        print(f"Epoch {epoch} loss = {loss.item()}")

if __name__ == "__main__":
    main()