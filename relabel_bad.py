#!/usr/bin/env python3
"""
relabel_bad.py — Smart Re-labeler for Close-up Bullseye Images
===============================================================
Problem with original labeler on close-up shots:
  - When target fills the frame, red mask merges center dot WITH rings
  - Largest contour becomes the whole target, not just the center dot

Fix strategy:
  - Instead of picking the LARGEST circular blob, we pick the
    SMALLEST blob that is still highly circular (circularity > 0.75)
    and has area above a minimum threshold.
  - This correctly isolates the solid center dot even when the
    outer rings are also red and present.

Affected source images:
  0005, 0006, 0007, 0008, 0009, 0012,
  0034, 0036, 0040, 0042, 0051, 0052
"""

import cv2
import numpy as np
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOURCE_DIR    = Path("/tmp/sparse_repo")
IMAGES_DIR    = Path("/home/claude/dataset/images/train")
LABELS_DIR    = Path("/home/claude/dataset/labels/train")
ANNOTATED_DIR = Path("/home/claude/dataset/annotated")

# ---------------------------------------------------------------------------
# HSV red ranges (same as before)
# ---------------------------------------------------------------------------
RED_LOWER1 = np.array([0,   120,  60])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([160, 120,  60])
RED_UPPER2 = np.array([180, 255, 255])

BBOX_PAD = 1.2

# ---------------------------------------------------------------------------
# Bad source image filenames
# ---------------------------------------------------------------------------
BAD_IMAGES = [
    "x_20260415_175154_0005.jpg",
    "x_20260415_175155_0006.jpg",
    "x_20260415_175156_0007.jpg",
    "x_20260415_175157_0008.jpg",
    "x_20260415_175158_0009.jpg",
    "x_20260415_175201_0012.jpg",
    "x_20260415_175224_0034.jpg",
    "x_20260415_175226_0036.jpg",
    "x_20260415_175230_0040.jpg",
    "x_20260415_175232_0042.jpg",
    "x_20260415_175241_0051.jpg",
    "x_20260415_175243_0052.jpg",
]

# ---------------------------------------------------------------------------
# Smart detection: pick MOST CIRCULAR blob in reasonable size range
# NOT the largest — to isolate the solid center dot only
# ---------------------------------------------------------------------------
def detect_center_dot_smart(img_bgr):
    H, W = img_bgr.shape[:2]
    img_area = H * W

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  k, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k, iterations=3)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Must be at least 300px² and at most 25% of image area
        if area < 300 or area > img_area * 0.25:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        # Must be reasonably circular
        if circularity < 0.50:
            continue
        candidates.append((circularity, area, cnt))

    if not candidates:
        # Fallback: relax circularity, pick most circular
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300 or area > img_area * 0.40:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            candidates.append((circularity, area, cnt))

    if not candidates:
        return None

    # Pick the MOST CIRCULAR candidate (not largest)
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, _, best = candidates[0]

    x, y, w, h = cv2.boundingRect(best)
    pad_w = int(w * (BBOX_PAD - 1) / 2)
    pad_h = int(h * (BBOX_PAD - 1) / 2)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = x2 - x1
    bh = y2 - y1
    return (cx, cy, bw, bh)


# ---------------------------------------------------------------------------
# Annotated preview drawing
# ---------------------------------------------------------------------------
def draw_annotated(img_bgr, cx, cy, bw, bh, img_w, img_h):
    vis = img_bgr.copy()
    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.line(vis, (int(cx) - 30, int(cy)), (int(cx) + 30, int(cy)), (0, 0, 255), 2)
    cv2.line(vis, (int(cx), int(cy) - 30), (int(cx), int(cy) + 30), (0, 0, 255), 2)
    cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    label = f"bullseye  cx={int(cx)}  cy={int(cy)}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(0, y1 - th - 6)
    cv2.rectangle(vis, (x1, ty), (x1 + tw + 6, y1), (0, 220, 0), -1)
    cv2.putText(vis, label, (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    cx_n = cx / img_w
    cy_n = cy / img_h
    bw_n = bw / img_w
    bh_n = bh / img_h

    bar = np.full((22, img_w, 3), (30, 30, 30), dtype=np.uint8)
    info = (f"  class=bullseye  cx={cx_n:.4f}  cy={cy_n:.4f}  "
            f"w={bw_n:.4f}  h={bh_n:.4f}  [YOLO normalized]")
    cv2.putText(bar, info, (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    vis = np.vstack([vis, bar])
    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print(f"[*] Re-labeling {len(BAD_IMAGES)} bad source images with smart detection")
    print("-" * 60)

    success = 0
    failed  = []

    for fname in BAD_IMAGES:
        src_path = SOURCE_DIR / fname
        if not src_path.exists():
            print(f"  [MISSING] {fname}")
            failed.append(fname)
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            print(f"  [SKIP] Cannot read: {fname}")
            failed.append(fname)
            continue

        H, W = img.shape[:2]
        result = detect_center_dot_smart(img)

        if result is None:
            print(f"  [FAIL] No detection: {fname}")
            failed.append(fname)
            continue

        cx, cy, bw, bh = result
        cx_n = cx / W
        cy_n = cy / H
        bw_n = bw / W
        bh_n = bh / H

        stem = Path(fname).stem

        # Save image to dataset
        shutil.copy2(src_path, IMAGES_DIR / fname)

        # Save YOLO label
        lbl_path = LABELS_DIR / (stem + ".txt")
        with open(lbl_path, "w") as f:
            f.write(f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")

        # Save annotated preview
        ann = draw_annotated(img, cx, cy, bw, bh, W, H)
        cv2.imwrite(str(ANNOTATED_DIR / fname), ann,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        print(f"  [OK] {fname:45s} center=({int(cx):4d},{int(cy):4d})  "
              f"box=({int(bw):3d}x{int(bh):3d})")
        success += 1

    print("-" * 60)
    print(f"[+] Re-labeled: {success}/{len(BAD_IMAGES)}")
    if failed:
        print(f"[!] Failed: {failed}")

    # Final counts
    n_img = len(list(IMAGES_DIR.glob("*.jpg")))
    n_lbl = len(list(LABELS_DIR.glob("*.txt")))
    print(f"\n[+] Dataset totals → images: {n_img}  labels: {n_lbl}")


if __name__ == "__main__":
    run()
