#!/usr/bin/env python3
"""
X_5_Bullseye.py — Red Bullseye Detector for Drone Precision Landing
=====================================================================
Adapted from X_5.py (green X target detector) for the red bullseye target.

Key changes from X_5.py:
  1. Default weights changed to best.pt (bullseye model)
  2. Detection box color changed from green to RED (matches target color convention)
  3. Crosshair drawn at exact bullseye center (cx, cy) on every detection
  4. Frame center crosshair always shown — so pilot/drone can see offset from center
  5. Offset vector printed: how far the bullseye is from frame center (dx, dy)
  6. Confidence threshold raised to 0.60 (model is strong enough at 0.995 mAP50)
  7. Landing alignment indicator: prints CENTERED if bullseye is within 50px of center
  8. All terminal output updated for bullseye context

Usage:
    python3 X_5_Bullseye.py --headless
    python3 X_5_Bullseye.py --headless --conf 0.6
    python3 X_5_Bullseye.py --snapshot
    python3 X_5_Bullseye.py
    python3 X_5_Bullseye.py --save-debug

Transfer to Jetson:
    scp X_5_Bullseye.py best.pt jetson@<ip>:~/
"""

import argparse
import time
import cv2
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Alignment threshold — how close (in pixels) center must be to declare ALIGNED
# ---------------------------------------------------------------------------
ALIGN_THRESHOLD_PX = 50   # within 50px of frame center = aligned for landing


# ---------------------------------------------------------------------------
# GStreamer pipeline (unchanged from X_5.py)
# ---------------------------------------------------------------------------
def build_gstreamer_pipeline(
    sensor_mode=0,
    capture_width=3840,
    capture_height=2160,
    framerate=30,
    display_width=1920,
    display_height=1080,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1, format=(string)NV12 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink max-buffers=2 drop=true"
    )


MODES = {
    "4k": {
        "sensor_mode": 0,
        "capture_width": 3840,
        "capture_height": 2160,
        "framerate": 30,
        "display_width": 1920,
        "display_height": 1080,
    },
    "1080p": {
        "sensor_mode": 1,
        "capture_width": 1920,
        "capture_height": 1080,
        "framerate": 60,
        "display_width": 1920,
        "display_height": 1080,
    },
}


# ---------------------------------------------------------------------------
# Load bullseye model
# ---------------------------------------------------------------------------
def load_model(weights="best.pt"):
    from ultralytics import YOLO

    if not os.path.exists(weights):
        print(f"[!] Model not found: {weights}")
        print(f"    scp best.pt jetson@<ip>:~/")
        exit(1)

    model = YOLO(weights)
    print(f"[+] Loaded bullseye model: {weights}")
    print(f"    Classes: {model.names}")
    return model


# ---------------------------------------------------------------------------
# Draw detections — RED box + crosshair at bullseye center + frame center guide
# ---------------------------------------------------------------------------
def draw_detections(frame, results, conf_thresh=0.60, scale_x=1.0, scale_y=1.0):
    """
    Returns (frame, detections, best_cx, best_cy)
    best_cx, best_cy = pixel coords of highest-conf bullseye center, or None
    """
    h, w = frame.shape[:2]
    frame_cx = w // 2
    frame_cy = h // 2

    # Always draw frame center crosshair in white
    cv2.drawMarker(frame, (frame_cx, frame_cy), (255, 255, 255),
                   cv2.MARKER_CROSS, 40, 1)

    detections = 0
    best_cx, best_cy = None, None
    best_conf = 0.0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            detections += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = result.names[int(box.cls[0])]

            # Remap to display coords
            dx1 = int(x1 * scale_x)
            dy1 = int(y1 * scale_y)
            dx2 = int(x2 * scale_x)
            dy2 = int(y2 * scale_y)
            dcx = (dx1 + dx2) // 2
            dcy = (dy1 + dy2) // 2

            # RED bounding box
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)

            # RED crosshair at bullseye center
            cv2.drawMarker(frame, (dcx, dcy), (0, 0, 255),
                           cv2.MARKER_CROSS, 50, 2)
            cv2.circle(frame, (dcx, dcy), 6, (0, 0, 255), -1)

            # Line from frame center to bullseye center (shows offset direction)
            cv2.line(frame, (frame_cx, frame_cy), (dcx, dcy), (0, 165, 255), 1)

            # Label: class + confidence
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (dx1, max(0, dy1 - th - 10)),
                          (dx1 + tw, dy1), (0, 0, 255), -1)
            cv2.putText(frame, label, (dx1, dy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Track best detection
            if conf > best_conf:
                best_conf = conf
                # Return in original (non-scaled) coords for accurate offset
                best_cx = (x1 + x2) // 2
                best_cy = (y1 + y2) // 2

    return frame, detections, best_cx, best_cy


# ---------------------------------------------------------------------------
# Alignment status: how far bullseye center is from frame center
# ---------------------------------------------------------------------------
def get_alignment_status(cx, cy, frame_w, frame_h):
    """
    Returns (dx, dy, aligned)
    dx = pixels to move RIGHT to center (negative = move LEFT)
    dy = pixels to move DOWN to center (negative = move UP)
    aligned = True if within ALIGN_THRESHOLD_PX
    """
    frame_cx = frame_w // 2
    frame_cy = frame_h // 2
    dx = cx - frame_cx   # positive = bullseye is to the RIGHT of center
    dy = cy - frame_cy   # positive = bullseye is BELOW center
    dist = (dx**2 + dy**2) ** 0.5
    aligned = dist <= ALIGN_THRESHOLD_PX
    return dx, dy, aligned


# ---------------------------------------------------------------------------
# Live mode (with display)
# ---------------------------------------------------------------------------
def run_live(args):
    print("[*] Loading bullseye detection model...")
    model = load_model(args.weights)

    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    print(f"[*] Opening camera: {args.mode} → "
          f"{mode['display_width']}x{mode['display_height']}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        print("    Check: ls /dev/video0")
        print("    Check: sudo systemctl restart nvargus-daemon")
        return

    view_w, view_h = 960, 540
    window_name = "Bullseye Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, view_w, view_h)

    fps_count = 0
    fps_start = time.time()
    display_fps = 0
    debug_saved = False

    print(f"[*] Running (conf={args.conf}, imgsz={args.imgsz}). "
          f"Press 'q' to quit, 's' to snapshot.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            if args.save_debug and not debug_saved:
                cv2.imwrite("debug_raw_frame.jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[DEBUG] Saved: debug_raw_frame.jpg "
                      f"({frame.shape[1]}x{frame.shape[0]})")
                debug_saved = True

            # Inference on full resolution
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            # Downscale for display AFTER inference
            display_frame = cv2.resize(frame, (view_w, view_h))
            scale_x = view_w / frame.shape[1]
            scale_y = view_h / frame.shape[0]

            display_frame, det_count, best_cx, best_cy = draw_detections(
                display_frame, results, args.conf, scale_x, scale_y
            )

            # FPS counter
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # Status bar
            if best_cx is not None:
                dx, dy, aligned = get_alignment_status(
                    best_cx, best_cy, frame.shape[1], frame.shape[0])
                status = "ALIGNED — READY TO LAND" if aligned else f"OFFSET  dx={dx:+d}px  dy={dy:+d}px"
                color = (0, 255, 0) if aligned else (0, 165, 255)
            else:
                status = "Searching for bullseye..."
                color = (0, 255, 255)

            info = f"FPS:{display_fps:.1f} | Bullseye:{det_count} | {status}"
            cv2.putText(display_frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                fname = f"bullseye_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[+] Saved: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Headless mode (SSH / no display)
# ---------------------------------------------------------------------------
def run_headless(args):
    print("[*] Loading bullseye detection model...")
    model = load_model(args.weights)

    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    print(f"[*] Opening camera: {args.mode} → "
          f"{mode['display_width']}x{mode['display_height']}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        print("    Check: ls /dev/video0")
        print("    Check: sudo systemctl restart nvargus-daemon")
        return

    fps_count = 0
    fps_start = time.time()
    display_fps = 0
    debug_saved = False

    print(f"[*] Headless mode (conf={args.conf}, imgsz={args.imgsz}). Ctrl+C to stop.")
    print(f"[*] Alignment threshold: ±{ALIGN_THRESHOLD_PX}px from frame center")
    print("-" * 70)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            if args.save_debug and not debug_saved:
                cv2.imwrite("debug_raw_frame.jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[DEBUG] Saved debug_raw_frame.jpg "
                      f"({frame.shape[1]}x{frame.shape[0]})")
                debug_saved = True

            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            frame_w, frame_h = frame.shape[1], frame.shape[0]
            det_count = 0
            best_cx, best_cy, best_conf = None, None, 0.0

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= args.conf:
                        det_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        if conf > best_conf:
                            best_conf = conf
                            best_cx, best_cy = cx, cy

            if best_cx is not None:
                dx, dy, aligned = get_alignment_status(best_cx, best_cy, frame_w, frame_h)
                align_str = "✓ ALIGNED" if aligned else f"dx={dx:+d} dy={dy:+d}"
                print(f"\r[BULLSEYE] conf={best_conf:.2f}  "
                      f"center=({best_cx},{best_cy})  "
                      f"{align_str}  "
                      f"FPS={display_fps:.1f}    ", end="", flush=True)
            else:
                print(f"\rSearching... FPS={display_fps:.1f} | No bullseye detected   ",
                      end="", flush=True)

            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

    except KeyboardInterrupt:
        print("\n[*] Stopped.")
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Snapshot mode — single frame, save annotated image
# ---------------------------------------------------------------------------
def run_snapshot(args):
    print("[*] Loading bullseye detection model...")
    model = load_model(args.weights)

    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        return

    print("[*] Warming up camera (30 frames)...")
    for _ in range(30):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[!] Failed to capture frame.")
        return

    results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
    annotated = frame.copy()
    annotated, det_count, best_cx, best_cy = draw_detections(
        annotated, results, args.conf)

    fname = f"bullseye_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(fname, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[+] Detected {det_count} bullseye(s)")
    print(f"[+] Saved: {fname} ({frame.shape[1]}x{frame.shape[0]})")

    if best_cx is not None:
        dx, dy, aligned = get_alignment_status(
            best_cx, best_cy, frame.shape[1], frame.shape[0])
        print(f"    Bullseye center: ({best_cx}, {best_cy})")
        print(f"    Frame center:    ({frame.shape[1]//2}, {frame.shape[0]//2})")
        print(f"    Offset:          dx={dx:+d}px  dy={dy:+d}px")
        print(f"    Status:          {'ALIGNED — READY TO LAND' if aligned else 'NOT ALIGNED'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Red Bullseye Detector — Jetson Nano + IMX477")
    parser.add_argument("--weights", default="best.pt",
                        help="Path to YOLO weights (default: best.pt)")
    parser.add_argument("--mode", choices=["4k", "1080p"], default="1080p",
                        help="Camera mode (default: 1080p)")
    parser.add_argument("--conf", type=float, default=0.60,
                        help="Confidence threshold (default: 0.60)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference size (default: 640)")
    parser.add_argument("--align-threshold", type=int, default=ALIGN_THRESHOLD_PX,
                        help=f"Pixel threshold for landing alignment (default: {ALIGN_THRESHOLD_PX})")
    parser.add_argument("--save-debug", action="store_true",
                        help="Save first raw frame for debugging")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--headless", action="store_true",
                       help="No display, terminal output only (SSH)")
    group.add_argument("--snapshot", action="store_true",
                       help="Single frame detection + save annotated image")

    args = parser.parse_args()
    ALIGN_THRESHOLD_PX = args.align_threshold

    if args.headless:
        run_headless(args)
    elif args.snapshot:
        run_snapshot(args)
    else:
        run_live(args)
