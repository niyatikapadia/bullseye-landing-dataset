#!/usr/bin/env python3
"""
auto_label.py — Red Bullseye Auto-Labeler
==========================================
Detects the solid red center dot of the bullseye in each image using
HSV color masking + contour analysis, then saves YOLO-format .txt labels.

Output structure:
    dataset/
        images/train/   ← copies of all .jpg files
        labels/train/   ← YOLO .txt label files
        data.yaml       ← YOLOv8 training config
    preview/            ← annotated preview images for verification

YOLO label format (per line):
    <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    All values normalized 0.0–1.0 relative to image dimensions.
    class_id = 0 (bullseye)

Usage:
    python3 auto_label.py
    python3 auto_label.py --images /path/to/images --preview
"""

import cv2
import numpy as np
import os
import shutil
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLASS_ID    = 0          # single class: bullseye
CLASS_NAME  = "bullseye"

# HSV range for the red center dot
# Red wraps around in HSV, so we need two ranges:
#   Lower red: H=0–10,   S=100–255, V=50–255
#   Upper red: H=160–180, S=100–255, V=50–255
RED_LOWER1 = np.array([0,   100,  50])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([160, 100,  50])
RED_UPPER2 = np.array([180, 255, 255])

# The center dot is the LARGEST solid red blob that is roughly circular
# Min area filter to exclude noise (in pixels, at original resolution)
MIN_AREA   = 500
MAX_AREA   = 400000   # won't flag the entire image if rings bleed together

# Bounding box padding multiplier (1.2 = 20% padding around detected blob)
BBOX_PAD   = 1.15

# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------
def detect_bullseye_center(img_bgr):
    """
    Returns (cx, cy, w, h) in pixel coordinates of the center dot bounding box,
    or None if not detected.
    
    Strategy:
    1. Convert to HSV
    2. Mask both red hue ranges
    3. Morphological clean-up (remove noise, fill holes)
    4. Find contours, pick the most circular one in a reasonable size range
    5. Return bounding box with padding
    """
    h_img, w_img = img_bgr.shape[:2]

    # Step 1: HSV conversion
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Step 2: Build red mask (both hue ranges combined)
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Step 3: Morphological ops — remove small noise, fill dot interior
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Step 4: Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    best = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        # Circularity = 4π·Area / Perimeter²  (1.0 = perfect circle)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Score = area × circularity  (prefer large, circular blobs)
        score = area * circularity
        if score > best_score:
            best_score = score
            best = cnt

    if best is None:
        return None

    # Step 5: Bounding box with padding
    x, y, w, h = cv2.boundingRect(best)

    # Apply padding symmetrically
    pad_w = int(w * (BBOX_PAD - 1) / 2)
    pad_h = int(h * (BBOX_PAD - 1) / 2)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = x2 - x1
    bh = y2 - y1

    return (cx, cy, bw, bh)


# ---------------------------------------------------------------------------
# YOLO label writer
# ---------------------------------------------------------------------------
def to_yolo(cx, cy, bw, bh, img_w, img_h):
    """Normalize pixel coords to YOLO format (0.0–1.0)."""
    return (
        cx  / img_w,
        cy  / img_h,
        bw  / img_w,
        bh  / img_h,
    )


# ---------------------------------------------------------------------------
# Preview drawing
# ---------------------------------------------------------------------------
def draw_preview(img_bgr, cx, cy, bw, bh, conf_label="AUTO"):
    vis = img_bgr.copy()
    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    # Bounding box
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # Center crosshair
    cv2.drawMarker(vis, (int(cx), int(cy)), (0, 0, 255),
                   cv2.MARKER_CROSS, 40, 3)
    # Label
    label = f"{CLASS_NAME} [{conf_label}]"
    cv2.putText(vis, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return vis


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(images_dir, output_dir, preview_dir, do_preview):
    images_dir  = Path(images_dir)
    output_dir  = Path(output_dir)
    preview_dir = Path(preview_dir)

    # Output folder structure
    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    if do_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # Collect all jpg images
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        print(f"[!] No .jpg images found in: {images_dir}")
        return

    print(f"[*] Found {len(image_paths)} images in {images_dir}")
    print(f"[*] Output  → {output_dir}")
    if do_preview:
        print(f"[*] Previews → {preview_dir}")
    print("-" * 60)

    success = 0
    failed  = []

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            failed.append(img_path.name)
            continue

        h_img, w_img = img_bgr.shape[:2]
        result = detect_bullseye_center(img_bgr)

        if result is None:
            print(f"  [FAIL] No bullseye detected: {img_path.name}")
            failed.append(img_path.name)
            continue

        cx, cy, bw, bh = result
        cx_n, cy_n, bw_n, bh_n = to_yolo(cx, cy, bw, bh, w_img, h_img)

        # Write YOLO label file
        label_path = train_lbl_dir / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write(f"{CLASS_ID} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")

        # Copy image to dataset folder
        dst_img = train_img_dir / img_path.name
        shutil.copy2(img_path, dst_img)

        # Save preview
        if do_preview:
            vis = draw_preview(img_bgr, cx, cy, bw, bh)
            # Downscale preview to 50% for easy viewing
            vis_small = cv2.resize(vis, (w_img // 2, h_img // 2))
            cv2.imwrite(str(preview_dir / img_path.name), vis_small)

        print(f"  [OK]   {img_path.name:40s} "
              f"center=({int(cx):4d},{int(cy):4d})  "
              f"box=({int(bw):3d}x{int(bh):3d})  "
              f"norm=({cx_n:.3f},{cy_n:.3f},{bw_n:.3f},{bh_n:.3f})")
        success += 1

    # Write data.yaml for YOLOv8 training
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/train\n\n")   # same set for now; expand later
        f.write(f"nc: 1\n")
        f.write(f"names: ['{CLASS_NAME}']\n")

    # Summary
    print("-" * 60)
    print(f"[+] Labeled:  {success}/{len(image_paths)} images")
    print(f"[+] Failed:   {len(failed)} images")
    if failed:
        print(f"    → {', '.join(failed)}")
    print(f"[+] Labels saved to:  {train_lbl_dir}")
    print(f"[+] Images copied to: {train_img_dir}")
    print(f"[+] data.yaml saved:  {yaml_path}")
    if do_preview:
        print(f"[+] Previews saved to: {preview_dir}")
    print()
    print("=" * 60)
    print("  NEXT STEP — Train on Google Colab:")
    print("  1. Zip the 'dataset/' folder and upload to Colab")
    print("  2. Run:")
    print("       from ultralytics import YOLO")
    print("       model = YOLO('yolov8n.pt')")
    print("       model.train(data='dataset/data.yaml',")
    print("                   epochs=100, imgsz=640,")
    print("                   batch=16, augment=True)")
    print("  3. Download runs/detect/train/weights/best.pt")
    print("  4. scp best.pt jetson@<ip>:~/")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Red Bullseye Auto-Labeler")
    parser.add_argument("--images",  default="/home/claude/jetson-camera",
                        help="Folder containing .jpg images (default: jetson-camera/)")
    parser.add_argument("--output",  default="/home/claude/dataset",
                        help="Output dataset folder (default: dataset/)")
    parser.add_argument("--preview", default="/home/claude/preview",
                        help="Preview output folder (default: preview/)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip saving preview images")
    args = parser.parse_args()

    run(
        images_dir  = args.images,
        output_dir  = args.output,
        preview_dir = args.preview,
        do_preview  = not args.no_preview,
    )
