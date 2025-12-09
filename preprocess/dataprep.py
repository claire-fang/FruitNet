import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry

IMG_DIR = os.path.join("2000images", "images")
LABEL_DIR = os.path.join("2000images", "labels")
OUT_ROOT = "masks_grabcut"
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"  # update with your checkpoint path
os.makedirs(OUT_ROOT, exist_ok=True)

anchor = set([2, 5, 6, 9, 14, 25, 37, 38, 39, 44, 45, 48, 49, 53, 55, 57, 61, 62, 63, 64, 66])


def build_predictor() -> SamPredictor:
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    return SamPredictor(sam)


def process_image(img_path: str, label_path: str, predictor: SamPredictor) -> int:
    """Generate SAM masks for a single image/label pair."""
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"skip {img_path}: unable to read image")
        return 0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    h, w = img_rgb.shape[:2]

    mask_count = 0
    with open(label_path, "r") as f:
        for row_idx, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])

            if cls_id not in anchor:
                continue
            cx_n, cy_n, bw_n, bh_n = map(float, parts[1:])

            cx = cx_n * w
            cy = cy_n * h
            bw = bw_n * w
            bh = bh_n * h

            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = min(w, max(0, x2))
            y2 = min(h, max(0, y2))

            rect_w = x2 - x1
            rect_h = y2 - y1
            if rect_w <= 0 or rect_h <= 0:
                continue

            box = np.array([x1, y1, x2 - 1, y2 - 1], dtype=np.float32)
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            if masks.size == 0:
                continue

            seg = (masks[0].astype("uint8") * 255)

            seg_bbox = np.zeros_like(seg)
            seg_bbox[y1:y2, x1:x2] = seg[y1:y2, x1:x2]

            class_dir = os.path.join(OUT_ROOT, str(cls_id))
            os.makedirs(class_dir, exist_ok=True)

            out_name = f"{image_id}*{row_idx}*{cls_id}.png"
            out_path = os.path.join(class_dir, out_name)

            cv2.imwrite(out_path, seg_bbox)
            mask_count += 1
    return mask_count


def main():

    image_files = sorted(glob(os.path.join(IMG_DIR, "*.jpg")))

    predictor = build_predictor()
    total_masks = 0
    for img_path in tqdm(image_files, desc="Processing images", unit="img"):
        label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, label_name)
        if not os.path.exists(label_path):
            print(f"skip {img_path}: label {label_name} not found")
            continue

        total_masks += process_image(img_path, label_path, predictor)

    print(f"done. generated {total_masks} masks from {len(image_files)} images.")


if __name__ == "__main__":
    main()
