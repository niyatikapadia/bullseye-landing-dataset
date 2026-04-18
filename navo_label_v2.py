#!/usr/bin/env python3
"""
navo_label_v2.py — Model-Based Outdoor Bullseye Labeler
========================================================
Uses the existing trained best.pt (mAP50=0.995) to label outdoor NAVO images.
This is more accurate than raw HSV detection because the model already knows
exactly what the bullseye looks like across varying conditions.

Strategy:
  - Run best.pt inference on each image at low confidence (0.25)
    so we catch even partially visible bullseyes
  - If detection found → save YOLO label
  - If no detection → save EMPTY label (background image)
    Background images are kept intentionally — they teach the model
    NOT to false-fire on outdoor scenes with no bullseye

Output:
  navo_dataset_v2/
    images/train/   ← all 213 images
    labels/train/   ← YOLO .txt (filled for detections, empty for backgrounds)
    annotated/      ← visual previews
    summary.txt     ← labeled vs background count
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOURCE_DIR    = Path("/home/claude/navo_raw/NAVO_DATA")
OUTPUT_DIR    = Path("/home/claude/navo_dataset_v2")
IMAGES_DIR    = OUTPUT_DIR / "images" / "train"
LABELS_DIR    = OUTPUT_DIR / "labels" / "train"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
MODEL_PATH    = "/home/claude/new_repo/model/best.pt"

# Low confidence threshold — catch even partial/difficult bullseyes
CONF_THRESHOLD = 0.25

for d in [IMAGES_DIR, LABELS_DIR, ANNOTATED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Draw annotated preview
# ---------------------------------------------------------------------------
def draw_annotated(img, detections, img_w, img_h):
    """detections = list of (cx, cy, bw, bh, conf) in pixel coords"""
    vis = img.copy()
    H, W = vis.shape[:2]

    if detections:
        for (cx, cy, bw, bh, conf) in detections:
            x1 = int(cx - bw/2); y1 = int(cy - bh/2)
            x2 = int(cx + bw/2); y2 = int(cy + bh/2)
            cv2.rectangle(vis, (x1,y1),(x2,y2),(0,220,0),2)
            cv2.line(vis,(int(cx)-25,int(cy)),(int(cx)+25,int(cy)),(0,0,255),2)
            cv2.line(vis,(int(cx),int(cy)-25),(int(cx),int(cy)+25),(0,0,255),2)
            cv2.circle(vis,(int(cx),int(cy)),5,(0,0,255),-1)
            label = f"bullseye {conf:.2f}"
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
            cv2.rectangle(vis,(x1,max(0,y1-th-8)),(x1+tw+6,y1),(0,220,0),-1)
            cv2.putText(vis,label,(x1+3,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1,cv2.LINE_AA)
        bar_color = (30,30,30)
        cx_n = detections[0][0]/img_w; cy_n = detections[0][1]/img_h
        bw_n = detections[0][2]/img_w; bh_n = detections[0][3]/img_h
        info = f"  LABELED  conf={detections[0][4]:.2f}  cx={cx_n:.4f} cy={cy_n:.4f} w={bw_n:.4f} h={bh_n:.4f}"
    else:
        bar_color = (0,0,150)
        info = "  BACKGROUND — no bullseye detected (empty label)"
        cv2.putText(vis,"BACKGROUND",(20,70),
                    cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,0,255),4,cv2.LINE_AA)

    bar = np.full((22,W,3),bar_color,dtype=np.uint8)
    cv2.putText(bar,info,(5,15),cv2.FONT_HERSHEY_SIMPLEX,
                0.42,(255,255,255),1,cv2.LINE_AA)
    return np.vstack([vis, bar])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print(f"[*] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[*] Classes: {model.names}")
    print(f"[*] Confidence threshold: {CONF_THRESHOLD}")

    images = sorted(SOURCE_DIR.glob("*.jpg"))
    total  = len(images)
    print(f"[*] Processing {total} outdoor NAVO images")
    print(f"[*] Output → {OUTPUT_DIR}")
    print("-" * 70)

    labeled    = 0
    background = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        H, W = img.shape[:2]

        # Run inference
        results = model(img, imgsz=640, conf=CONF_THRESHOLD, verbose=False)

        detections = []
        best_conf  = 0.0
        best_det   = None

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx = (x1+x2)/2; cy = (y1+y2)/2
                bw = x2-x1;     bh = y2-y1
                detections.append((cx, cy, bw, bh, conf))
                if conf > best_conf:
                    best_conf = conf
                    best_det  = (cx, cy, bw, bh, conf)

        # Copy image
        shutil.copy2(img_path, IMAGES_DIR / img_path.name)

        stem     = img_path.stem
        lbl_path = LABELS_DIR / (stem + ".txt")

        if best_det is not None:
            cx, cy, bw, bh, conf = best_det
            cx_n = cx/W; cy_n = cy/H; bw_n = bw/W; bh_n = bh/H
            with open(lbl_path, "w") as f:
                f.write(f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")
            labeled += 1
            status = f"LABELED   conf={conf:.2f}  center=({int(cx):4d},{int(cy):4d})"
        else:
            open(lbl_path, "w").close()
            background += 1
            status = "BACKGROUND (empty label)"

        # Annotated preview — downscale to 50% for storage
        ann = draw_annotated(img, [best_det] if best_det else [], W, H)
        ann_small = cv2.resize(ann, (W//2, (H+22)//2))
        cv2.imwrite(str(ANNOTATED_DIR / img_path.name), ann_small,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        print(f"  {img_path.name:45s} {status}")

    # data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR.resolve()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/train\n\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['bullseye']\n")

    summary = f"Total: {total}\nLabeled: {labeled}\nBackground: {background}\n"
    (OUTPUT_DIR / "summary.txt").write_text(summary)

    print("-" * 70)
    print(f"[+] Labeled:    {labeled}/{total}")
    print(f"[+] Background: {background}/{total}")
    print(f"[+] Output:     {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
