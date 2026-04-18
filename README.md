# Red Bullseye Landing Dataset — YOLOv8 Drone Precision Landing

Detection pipeline for a red bullseye target used in drone autonomous precision landing.
Built for **Jetson Nano + IMX477 camera + Pixhawk 6C** running ArduCopter.

---

## Model Performance

| Metric | Score |
|--------|-------|
| mAP50 | **0.995** (99.5%) |
| mAP50-95 | **0.953** (95.3%) |
| Precision | **0.999** (99.9%) |
| Recall | **1.000** (100%) |
| Inference speed | 1.9ms/frame (T4 GPU) |
| Model size | 6.2MB |

---

## Repo Structure

```
original_images/                ← 52 raw captured frames from Jetson camera
dataset/
  images/train/                 ← 911 images (47 originals × 15 augmentations)
  labels/train/                 ← 911 YOLO .txt label files (class 0 = bullseye)
  annotated/                    ← 911 visually marked preview images
  data.yaml                     ← YOLOv8 training config
model/
  best.pt                       ← Trained YOLOv8n model (mAP50=0.995)
X_5_bullseye.py                 ← Main detection script for Jetson deployment
x_detect_guide_bullseye.py      ← Detection + visual centering guide
x_detect_motor_bullseye.py      ← Detection + pymavlink motor control
x_detect_mavsdk_bullseye.py     ← Detection + MAVSDK offboard control
x_bench_test_bullseye.py        ← Bench test: motor response to bullseye offset
x_bench_fakegps.py              ← Fake GPS injection for bench testing
porter.py                       ← Pixhawk serial port inspector
train_bullseye_yolov8.ipynb     ← Google Colab training notebook
auto_label.py                   ← Auto-labeling (HSV color detection)
augment_clean.py                ← 15-type augmentation pipeline
annotate_dataset.py             ← Visual annotation renderer
relabel_bad.py                  ← Smart re-labeler for difficult shots
requirements.txt                ← Python dependencies
```

---

## First-Time Deployment (Jetson connected to monitor via HDMI)

Follow these steps in order on the Jetson terminal.

**Step 1 — Check the Jetson's IP address:**
```bash
hostname -I
```

**Step 2 — Clone the repo:**
```bash
cd ~
git clone https://github.com/niyatikapadia/bullseye-landing-dataset.git
cd bullseye-landing-dataset
```

**Step 3 — Copy the model to the working directory:**
```bash
cp model/best.pt .
```

**Step 4 — Install dependencies:**
```bash
pip install ultralytics pymavlink mavsdk opencv-python --break-system-packages
```

**Step 5 — Verify the camera is detected:**
```bash
ls /dev/video*
```

**Step 6 — Restart the camera daemon (run this if camera fails to open):**
```bash
sudo systemctl restart nvargus-daemon
```

**Step 7 — Run the detector:**

Headless mode (terminal output only, no window):
```bash
python3 X_5_bullseye.py --headless --weights best.pt --conf 0.6
```

With live display window on the HDMI monitor:
```bash
python3 X_5_bullseye.py --weights best.pt --conf 0.6
```

Single snapshot (saves annotated image to disk):
```bash
python3 X_5_bullseye.py --snapshot --weights best.pt --conf 0.6
```

---

## Subsequent Runs

Once installed, you only need:
```bash
cd ~/bullseye-landing-dataset
python3 X_5_bullseye.py --headless --weights best.pt --conf 0.6
```

To pull the latest updates from the repo:
```bash
cd ~/bullseye-landing-dataset
git pull
cp model/best.pt .
```

---

## Detection Scripts

| Script | Use case |
|--------|----------|
| `X_5_bullseye.py` | Main — detection + alignment guidance |
| `x_detect_guide_bullseye.py` | Visual arrows + direction guide on display |
| `x_detect_motor_bullseye.py` | Arms motors when bullseye detected (REMOVE PROPS FIRST) |
| `x_detect_mavsdk_bullseye.py` | MAVSDK offboard control on detection |
| `x_bench_test_bullseye.py` | Bench test — offset mapped to motor differential |

To run any other script:
```bash
python3 <script_name> --weights best.pt --conf 0.6
```

For motor control scripts, always test with `--dry-run` first:
```bash
python3 x_detect_motor_bullseye.py --weights best.pt --conf 0.6 --dry-run
```

---

## Training

Open `train_bullseye_yolov8.ipynb` in Google Colab, select T4 GPU, run all cells.
Dataset clones directly from this repo — no manual upload needed.

---

## Dataset Pipeline

```
52 raw images → auto_label.py → relabel_bad.py → augment_clean.py (×15) → 911 images → best.pt
```

**15 augmentation types:** hflip, vflip, rot90, rot180, blur, brightness, hsv, noise, shadow, scale_up, scale_down, crop, gamma, combined_a, combined_b

---

## Class

| ID | Name | Description |
|----|------|-------------|
| 0 | `bullseye` | Solid red center dot of the bullseye target |

---

## Install Dependencies

```bash
pip install -r requirements.txt
```
