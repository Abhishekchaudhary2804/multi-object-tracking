#!/usr/bin/env python3
# =============================================================
#  test_detect.py  —  Run this FIRST to verify detection works
#  Usage: python test_detect.py
# =============================================================

import cv2
import sys
import os

# ─── Check video exists ───────────────────────────────────────
VIDEO = "input/video.mp4"

if not os.path.exists(VIDEO):
    print(f"\n[ERROR] No video found at: {VIDEO}")
    print("  Steps to fix:")
    print("  1. Run: pip install yt-dlp")
    print("  2. Run: yt-dlp -f \"bestvideo[height<=720]+bestaudio\" \\")
    print("              -o \"input/video.%(ext)s\" YOUR_YOUTUBE_URL\n")
    sys.exit(1)

print("\n[Test] Running quick detection check …")
print("       (This will auto-download yolov8m.pt on first run)\n")

# ─── Load model ───────────────────────────────────────────────
from ultralytics import YOLO
model = YOLO("yolov8m.pt")

# ─── Read first frame ─────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
cap.release()

if not ret:
    print("[ERROR] Could not read first frame from video.")
    sys.exit(1)

print(f"[Test] Frame size: {frame.shape[1]}×{frame.shape[0]}")

# ─── Detect ───────────────────────────────────────────────────
results = model.predict(frame, conf=0.3, classes=[0], verbose=True)[0]
n = len(results.boxes)
print(f"\n[Test] ✅  Detected {n} person(s) on frame 1\n")

# ─── Save annotated preview ───────────────────────────────────
annotated = results.plot()
os.makedirs("screenshots", exist_ok=True)
out = "screenshots/detection_test.jpg"
cv2.imwrite(out, annotated)
print(f"[Test] Preview saved → {out}")
print("       Open that file to visually verify detection quality.\n")

if n == 0:
    print("[HINT] No people detected. Try:")
    print("       • Lower confidence: model.predict(..., conf=0.2)")
    print("       • Use a different frame (seek to 5 seconds in)")
    print("       • Check that your video actually has people in it\n")
