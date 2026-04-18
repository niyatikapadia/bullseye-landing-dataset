#!/usr/bin/env python3
"""
annotate_dataset.py — Visual Annotation Renderer
=================================================
Reads every image + its YOLO .txt label and burns in:
  - Green bounding box around the bullseye center dot
  - Red crosshair at the exact center (cx, cy)
  - Label text: "bullseye  cx=XXX  cy=XXX  conf=AUTO"
  - Small info bar at bottom: image name + normalized coords

Output:
    dataset/annotated/   ← annotated versions of ALL images
                           (original + augmented)

Usage:
    python3 annotate_dataset.py
"""

import cv2
import numpy as np
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMAGES_DIR    = Path("/home/claude/dataset/images/train")
LABELS_DIR    = Path("/home/claude/dataset/labels/train")
ANNOTATED_DIR = Path("/home/claude/dataset/annotated")

ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style config
# ---------------------------------------------------------------------------
BOX_COLOR        = (0, 220, 0)       # green bounding box
CROSS_COLOR      = (0, 0, 255)       # red crosshair
LABEL_BG_COLOR   = (0, 220, 0)       # green label background
LABEL_TEXT_COLOR = (0, 0, 0)         # black text on label
INFO_BG_COLOR    = (30, 30, 30)      # dark bar at bottom
INFO_TEXT_COLOR  = (255, 255, 255)   # white info text

BOX_THICKNESS    = 2
CROSS_SIZE       = 30                # crosshair arm length in pixels
CROSS_THICKNESS  = 2
FONT             = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.55
INFO_FONT_SCALE  = 0.42

CLASS_NAMES = {0: "bullseye"}

# ---------------------------------------------------------------------------
# Core drawing function
# ---------------------------------------------------------------------------
def annotate_image(img_bgr, labels):
    """
    img_bgr : numpy array (H, W, 3)
    labels  : list of (class_id, cx_n, cy_n, w_n, h_n) — normalized
    Returns annotated copy.
    """
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    for (cls_id, cx_n, cy_n, w_n, h_n) in labels:

        # Convert normalized → pixel coords
        cx  = int(cx_n * W)
        cy  = int(cy_n * H)
        bw  = int(w_n  * W)
        bh  = int(h_n  * H)

        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(W, cx + bw // 2)
        y2 = min(H, cy + bh // 2)

        cls_name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")

        # --- Bounding box ---
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # --- Crosshair at center ---
        cv2.line(vis,
                 (cx - CROSS_SIZE, cy), (cx + CROSS_SIZE, cy),
                 CROSS_COLOR, CROSS_THICKNESS)
        cv2.line(vis,
                 (cx, cy - CROSS_SIZE), (cx, cy + CROSS_SIZE),
                 CROSS_COLOR, CROSS_THICKNESS)
        # Small filled circle at exact center
        cv2.circle(vis, (cx, cy), 4, CROSS_COLOR, -1)

        # --- Label tag above box ---
        label_text = f"{cls_name}  cx={cx}  cy={cy}"
        (tw, th), baseline = cv2.getTextSize(
            label_text, FONT, LABEL_FONT_SCALE, 1)
        tag_y1 = max(0, y1 - th - 8)
        tag_y2 = y1
        tag_x2 = min(W, x1 + tw + 6)

        # Label background
        cv2.rectangle(vis, (x1, tag_y1), (tag_x2, tag_y2),
                      LABEL_BG_COLOR, -1)
        # Label text
        cv2.putText(vis, label_text,
                    (x1 + 3, tag_y2 - 3),
                    FONT, LABEL_FONT_SCALE, LABEL_TEXT_COLOR, 1,
                    cv2.LINE_AA)

        # --- Corner dot markers on bounding box ---
        dot_r = 4
        for (px, py) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            cv2.circle(vis, (px, py), dot_r, BOX_COLOR, -1)

    # --- Bottom info bar ---
    bar_h = 22
    bar   = np.full((bar_h, W, 3), INFO_BG_COLOR, dtype=np.uint8)

    if labels:
        cls_id, cx_n, cy_n, w_n, h_n = labels[0]
        info = (f"  class={CLASS_NAMES.get(cls_id,'?')}  "
                f"cx={cx_n:.4f}  cy={cy_n:.4f}  "
                f"w={w_n:.4f}  h={h_n:.4f}  "
                f"[YOLO normalized]")
    else:
        info = "  NO LABEL"

    cv2.putText(bar, info, (5, bar_h - 5),
                FONT, INFO_FONT_SCALE, INFO_TEXT_COLOR, 1, cv2.LINE_AA)

    vis = np.vstack([vis, bar])
    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    total = len(image_paths)
    print(f"[*] Annotating {total} images...")
    print(f"[*] Output → {ANNOTATED_DIR}")
    print("-" * 60)

    success = 0
    no_label = 0

    for i, img_path in enumerate(image_paths):
        label_path = LABELS_DIR / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        # Parse YOLO label file
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(float(parts[0]))
                        cx_n, cy_n, w_n, h_n = map(float, parts[1:])
                        labels.append((cls_id, cx_n, cy_n, w_n, h_n))
        else:
            no_label += 1

        # Draw annotations
        annotated = annotate_image(img, labels)

        # Save — keep original filename so it maps 1:1
        out_path = ANNOTATED_DIR / img_path.name
        cv2.imwrite(str(out_path), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        success += 1
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1:>4}/{total}] done...")

    print("-" * 60)
    print(f"[+] Annotated:  {success}/{total} images")
    print(f"[+] No label:   {no_label} images")
    print(f"[+] Saved to:   {ANNOTATED_DIR}")


if __name__ == "__main__":
    run()
