#!/usr/bin/env python3
"""
augment_clean.py — Heavy Augmentation Pipeline
================================================
Runs on the 47 clean source originals only (no _aug_ files).
Generates ~15 augmentations per image → ~700 total images.

Augmentation types:
  1.  hflip          — horizontal flip
  2.  vflip          — vertical flip
  3.  rot90          — 90° rotation
  4.  rot180         — 180° rotation
  5.  blur           — gaussian blur (simulate drone motion)
  6.  brightness     — random brightness/contrast
  7.  hsv            — hue/saturation shift (lighting variation)
  8.  noise          — gaussian noise
  9.  shadow         — random shadow overlay
  10. scale_up       — zoom in (target larger)
  11. scale_down     — zoom out (target smaller)
  12. crop           — random crop & resize
  13. gamma          — gamma correction
  14. combined_a     — flip + brightness + blur
  15. combined_b     — hsv + noise + scale

All bbox transforms are handled correctly so labels stay accurate.
"""

import cv2
import numpy as np
import random
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMAGES_DIR    = Path("/home/claude/dataset/images/train")
LABELS_DIR    = Path("/home/claude/dataset/labels/train")
ANNOTATED_DIR = Path("/home/claude/dataset/annotated")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_label(label_path):
    """Returns list of (cls, cx_n, cy_n, w_n, h_n)"""
    labels = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append((int(float(parts[0])),
                                   float(parts[1]), float(parts[2]),
                                   float(parts[3]), float(parts[4])))
    return labels

def write_label(label_path, labels):
    with open(label_path, "w") as f:
        for (cls, cx_n, cy_n, w_n, h_n) in labels:
            f.write(f"{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def draw_annotated(img, labels):
    """Burn bounding box + crosshair into image for preview."""
    vis = img.copy()
    H, W = vis.shape[:2]
    for (cls, cx_n, cy_n, w_n, h_n) in labels:
        cx = int(cx_n * W); cy = int(cy_n * H)
        bw = int(w_n  * W); bh = int(h_n  * H)
        x1 = max(0, cx - bw//2); y1 = max(0, cy - bh//2)
        x2 = min(W, cx + bw//2); y2 = min(H, cy + bh//2)
        cv2.rectangle(vis, (x1,y1),(x2,y2),(0,220,0),2)
        cv2.line(vis,(cx-25,cy),(cx+25,cy),(0,0,255),2)
        cv2.line(vis,(cx,cy-25),(cx,cy+25),(0,0,255),2)
        cv2.circle(vis,(cx,cy),4,(0,0,255),-1)
        lbl = f"bullseye cx={cx} cy={cy}"
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        cv2.rectangle(vis,(x1,max(0,y1-th-6)),(x1+tw+6,y1),(0,220,0),-1)
        cv2.putText(vis,lbl,(x1+3,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
    bar = np.full((20,W,3),(30,30,30),dtype=np.uint8)
    if labels:
        _,cx_n,cy_n,w_n,h_n = labels[0]
        info = f"  cx={cx_n:.4f} cy={cy_n:.4f} w={w_n:.4f} h={h_n:.4f} [YOLO norm]"
        cv2.putText(bar,info,(5,14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1,cv2.LINE_AA)
    return np.vstack([vis,bar])

def save_aug(img, labels, stem, aug_name):
    fname = f"{stem}_aug_{aug_name}.jpg"
    out_img  = IMAGES_DIR    / fname
    out_lbl  = LABELS_DIR    / (stem + f"_aug_{aug_name}.txt")
    out_ann  = ANNOTATED_DIR / fname
    cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    write_label(out_lbl, labels)
    ann = draw_annotated(img, labels)
    cv2.imwrite(str(out_ann), ann, [cv2.IMWRITE_JPEG_QUALITY, 90])

# ---------------------------------------------------------------------------
# Augmentation functions — each returns (aug_img, aug_labels)
# Labels are YOLO normalized (cx_n, cy_n, w_n, h_n) in [0,1]
# ---------------------------------------------------------------------------

def aug_hflip(img, labels):
    out = cv2.flip(img, 1)
    new = [(c, clamp(1.0 - cx), cy, w, h) for c,cx,cy,w,h in labels]
    return out, new

def aug_vflip(img, labels):
    out = cv2.flip(img, 0)
    new = [(c, cx, clamp(1.0 - cy), w, h) for c,cx,cy,w,h in labels]
    return out, new

def aug_rot90(img, labels):
    # Rotate 90° clockwise: (cx,cy) → (1-cy, cx)
    out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    new = [(c, clamp(1.0-cy), clamp(cx), h, w) for c,cx,cy,w,h in labels]
    return out, new

def aug_rot180(img, labels):
    out = cv2.rotate(img, cv2.ROTATE_180)
    new = [(c, clamp(1.0-cx), clamp(1.0-cy), w, h) for c,cx,cy,w,h in labels]
    return out, new

def aug_blur(img, labels):
    k = random.choice([5, 7, 9, 11])
    out = cv2.GaussianBlur(img, (k, k), 0)
    return out, labels

def aug_brightness(img, labels):
    alpha = random.uniform(0.6, 1.5)   # contrast
    beta  = random.randint(-40, 40)     # brightness
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out, labels

def aug_hsv(img, labels):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:,:,0] = np.clip(hsv[:,:,0] + random.randint(-10, 10), 0, 179)
    hsv[:,:,1] = np.clip(hsv[:,:,1] + random.randint(-40, 40), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + random.randint(-30, 30), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out, labels

def aug_noise(img, labels):
    noise = np.random.normal(0, random.uniform(5, 20), img.shape).astype(np.int16)
    out = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out, labels

def aug_shadow(img, labels):
    """Add a random dark polygon shadow over part of the image."""
    out = img.copy()
    H, W = out.shape[:2]
    # Random shadow polygon (2-4 vertices)
    n_pts = random.randint(3, 5)
    pts = np.array([[random.randint(0,W), random.randint(0,H)] for _ in range(n_pts)], np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    alpha = random.uniform(0.35, 0.60)
    shadow = out.copy()
    shadow[mask==255] = (shadow[mask==255] * alpha).astype(np.uint8)
    out = shadow
    return out, labels

def aug_scale_up(img, labels):
    """Zoom in — crop center 80% and resize back."""
    H, W = img.shape[:2]
    scale = random.uniform(0.70, 0.85)
    dw = int(W * (1 - scale) / 2)
    dh = int(H * (1 - scale) / 2)
    cropped = img[dh:H-dh, dw:W-dw]
    out = cv2.resize(cropped, (W, H))
    # Remap labels: new_cx = (old_cx - dw/W) / scale
    new = []
    for c,cx,cy,w,h in labels:
        ncx = clamp((cx - dw/W) / scale)
        ncy = clamp((cy - dh/H) / scale)
        nw  = clamp(w / scale)
        nh  = clamp(h / scale)
        new.append((c, ncx, ncy, nw, nh))
    return out, new

def aug_scale_down(img, labels):
    """Zoom out — paste image on gray canvas smaller."""
    H, W = img.shape[:2]
    scale = random.uniform(0.55, 0.78)
    new_w = int(W * scale)
    new_h = int(H * scale)
    small = cv2.resize(img, (new_w, new_h))
    # Paste offset
    ox = random.randint(0, W - new_w)
    oy = random.randint(0, H - new_h)
    canvas = np.full((H, W, 3), 128, dtype=np.uint8)
    canvas[oy:oy+new_h, ox:ox+new_w] = small
    new = []
    for c,cx,cy,w,h in labels:
        ncx = clamp(ox/W + cx*scale)
        ncy = clamp(oy/H + cy*scale)
        nw  = clamp(w * scale)
        nh  = clamp(h * scale)
        new.append((c, ncx, ncy, nw, nh))
    return canvas, new

def aug_crop(img, labels):
    """Random crop keeping the target inside."""
    H, W = img.shape[:2]
    if not labels:
        return img, labels
    c,cx,cy,w,h = labels[0]
    # Target pixel bounds
    tx1 = cx - w/2; tx2 = cx + w/2
    ty1 = cy - h/2; ty2 = cy + h/2
    # Crop bounds must contain target with margin
    margin = 0.05
    x1_max = max(0.0, tx1 - margin)
    y1_max = max(0.0, ty1 - margin)
    x2_min = min(1.0, tx2 + margin)
    y2_min = min(1.0, ty2 + margin)
    if x1_max >= x2_min or y1_max >= y2_min:
        return img, labels
    x1_n = random.uniform(0.0, x1_max)
    y1_n = random.uniform(0.0, y1_max)
    x2_n = random.uniform(x2_min, 1.0)
    y2_n = random.uniform(y2_min, 1.0)
    x1p = int(x1_n*W); y1p = int(y1_n*H)
    x2p = int(x2_n*W); y2p = int(y2_n*H)
    if x2p <= x1p or y2p <= y1p:
        return img, labels
    cropped = img[y1p:y2p, x1p:x2p]
    out = cv2.resize(cropped, (W, H))
    cw = x2_n - x1_n; ch = y2_n - y1_n
    ncx = clamp((cx - x1_n) / cw)
    ncy = clamp((cy - y1_n) / ch)
    nw  = clamp(w / cw)
    nh  = clamp(h / ch)
    return out, [(c, ncx, ncy, nw, nh)]

def aug_gamma(img, labels):
    gamma = random.uniform(0.5, 1.8)
    inv_g = 1.0 / gamma
    table = np.array([(i/255.0)**inv_g * 255 for i in range(256)]).astype(np.uint8)
    out = cv2.LUT(img, table)
    return out, labels

def aug_combined_a(img, labels):
    img, labels = aug_hflip(img, labels)
    img, labels = aug_brightness(img, labels)
    img, labels = aug_blur(img, labels)
    return img, labels

def aug_combined_b(img, labels):
    img, labels = aug_hsv(img, labels)
    img, labels = aug_noise(img, labels)
    img, labels = aug_scale_down(img, labels)
    return img, labels

# ---------------------------------------------------------------------------
# All augmentation steps
# ---------------------------------------------------------------------------
AUGMENTATIONS = [
    ("hflip",      aug_hflip),
    ("vflip",      aug_vflip),
    ("rot90",      aug_rot90),
    ("rot180",     aug_rot180),
    ("blur",       aug_blur),
    ("brightness", aug_brightness),
    ("hsv",        aug_hsv),
    ("noise",      aug_noise),
    ("shadow",     aug_shadow),
    ("scale_up",   aug_scale_up),
    ("scale_down", aug_scale_down),
    ("crop",       aug_crop),
    ("gamma",      aug_gamma),
    ("combined_a", aug_combined_a),
    ("combined_b", aug_combined_b),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    # Only augment source originals (no _aug_ files)
    source_images = sorted([
        p for p in IMAGES_DIR.glob("*.jpg")
        if "_aug_" not in p.name
    ])

    print(f"[*] Source originals found: {len(source_images)}")
    print(f"[*] Augmentations per image: {len(AUGMENTATIONS)}")
    print(f"[*] Expected new images: {len(source_images) * len(AUGMENTATIONS)}")
    print(f"[*] Expected total: {len(source_images) * (len(AUGMENTATIONS)+1)}")
    print("-" * 60)

    generated = 0
    skipped   = 0

    for img_path in source_images:
        stem = img_path.stem
        lbl_path = LABELS_DIR / (stem + ".txt")
        labels = read_label(lbl_path)

        if not labels:
            print(f"  [SKIP - no label] {img_path.name}")
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP - bad img] {img_path.name}")
            skipped += 1
            continue

        for aug_name, aug_fn in AUGMENTATIONS:
            out_fname = f"{stem}_aug_{aug_name}.jpg"
            # Skip if already exists
            if (IMAGES_DIR / out_fname).exists():
                generated += 1
                continue
            try:
                aug_img, aug_labels = aug_fn(img.copy(), list(labels))
                if aug_labels:
                    save_aug(aug_img, aug_labels, stem, aug_name)
                    generated += 1
            except Exception as e:
                print(f"  [ERROR] {stem} {aug_name}: {e}")

        print(f"  [OK] {img_path.name}")

    print("-" * 60)
    n_img = len(list(IMAGES_DIR.glob("*.jpg")))
    n_lbl = len(list(LABELS_DIR.glob("*.txt")))
    print(f"[+] New augmentations generated: {generated}")
    print(f"[+] Skipped: {skipped}")
    print(f"[+] Total images in dataset: {n_img}")
    print(f"[+] Total labels in dataset: {n_lbl}")


if __name__ == "__main__":
    run()
