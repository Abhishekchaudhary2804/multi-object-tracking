#!/usr/bin/env python3
# =============================================================
#  main.py  —  Multi-Object Detection & Persistent ID Tracking
#  Run: python main.py
#  Run test (30s only): python main.py --test
# =============================================================

import argparse
import cv2
import sys
import os
import time

import config
from detector   import PersonDetector
from tracker    import PersistentTracker, TrailManager
from visualizer import draw_tracks, draw_hud
from utils      import (
    get_video_info, print_video_info,
    make_video_writer, HeatmapAccumulator,
    save_count_plot, ProgressBar,
    save_screenshot, ensure_dirs,
)


# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Sports Multi-Object Tracker")
    p.add_argument("--video",  default=config.VIDEO_PATH,
                   help="Path to input video")
    p.add_argument("--output", default=config.OUTPUT_PATH,
                   help="Path for annotated output video")
    p.add_argument("--test",   action="store_true",
                   help="Quick test: process only first 300 frames")
    p.add_argument("--no-trails", action="store_true",
                   help="Disable movement trails")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
def run_pipeline(video_path: str, output_path: str,
                 max_frames=None, show_trails=True):
    """
    Full detection + tracking + annotation pipeline.
    """

    # ── Validate video ────────────────────────────────────────
    if not os.path.exists(video_path):
        print(f"\n[ERROR] Video not found: {video_path}")
        print("  → Put your downloaded video in the 'input/' folder")
        print("  → Or update VIDEO_PATH in config.py\n")
        sys.exit(1)

    # ── Video info ────────────────────────────────────────────
    info = get_video_info(video_path)
    print_video_info(info)

    W   = info["width"]
    H   = info["height"]
    FPS = info["fps"]
    TOTAL = min(info["total_frames"], max_frames or 10**9)

    # ── Ensure output dirs ────────────────────────────────────
    ensure_dirs("output", "screenshots")

    # ── Initialise components ─────────────────────────────────
    print("[Main] Initialising detector …")
    detector = PersonDetector(
        model_path=config.MODEL_SIZE,
        confidence=config.CONFIDENCE,
        iou=config.IOU_THRESHOLD,
        classes=config.CLASSES,
        device=config.DEVICE,
    )

    print("[Main] Initialising tracker …")
    tracker = PersistentTracker(
        track_thresh=config.TRACK_THRESH,
        track_buffer=config.TRACK_BUFFER,
        match_thresh=config.MATCH_THRESH,
    )
    trails  = TrailManager(max_len=config.TRAIL_LENGTH)
    heatmap = HeatmapAccumulator(W, H)

    # ── Video I/O ─────────────────────────────────────────────
    cap    = cv2.VideoCapture(video_path)
    writer = make_video_writer(output_path, FPS, W, H)

    # ── Stats ─────────────────────────────────────────────────
    count_history = []       # active tracks per frame
    all_ids_seen  = set()
    screenshot_frames = {int(TOTAL * 0.1), int(TOTAL * 0.4),
                         int(TOTAL * 0.7), int(TOTAL * 0.95)}
    frame_times   = []

    progress = ProgressBar(TOTAL)
    print(f"\n[Main] Processing {TOTAL} frames → {output_path}\n")

    frame_n = 0

    # ── Main loop ─────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_n >= TOTAL:
            break

        t0 = time.time()

        # Skip frames if configured
        if frame_n % config.FRAME_SKIP == 0:
            # 1. Detect
            detections = detector.detect(frame)

            # 2. Track
            tracked = tracker.update(detections, frame.shape)

            # 3. Update trail history & heatmap
            active_ids = 0
            try:
                if tracked is not None and len(tracked) > 0 and tracked.tracker_id is not None:
                    active_ids = len(tracked.tracker_id)
                    for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
                        cx = int((xyxy[0] + xyxy[2]) / 2)
                        cy = int(xyxy[3])           # foot point
                        trails.update(int(tid), cx, cy)
                        all_ids_seen.add(int(tid))
                    heatmap.update(tracked.xyxy)
            except AttributeError:
                # DeepSORT path
                active_ids = len([t for t in tracked if t.is_confirmed()])

            count_history.append(active_ids)

            # 4. Draw
            trail_data = trails.get_all() if show_trails else None
            frame = draw_tracks(
                frame, tracked, trails=trail_data,
                show_confidence=config.SHOW_CONFIDENCE,
                box_thickness=config.BOX_THICKNESS,
                font_scale=config.FONT_SCALE,
            )

            # 5. HUD
            elapsed = time.time() - t0
            fps_live = 1.0 / max(elapsed, 1e-6)
            frame = draw_hud(frame, frame_n, fps_live, active_ids)

            # 6. Auto-screenshots
            if frame_n in screenshot_frames:
                shot_path = f"screenshots/frame_{frame_n:05d}.jpg"
                save_screenshot(frame, shot_path)

        # Write frame
        writer.write(frame)
        frame_n += 1
        frame_times.append(time.time() - t0)
        progress.update()

    # ── Cleanup ───────────────────────────────────────────────
    progress.done()
    cap.release()
    writer.release()

    # ── Save extras ───────────────────────────────────────────
    heatmap.save(config.HEATMAP_PATH)
    save_count_plot(count_history, FPS, config.COUNT_PLOT)

    # ── Summary ───────────────────────────────────────────────
    avg_fps = 1.0 / max(sum(frame_times) / max(len(frame_times), 1), 1e-6)
    print("\n" + "═" * 45)
    print("  PIPELINE COMPLETE")
    print("═" * 45)
    print(f"  Frames processed : {frame_n}")
    print(f"  Unique IDs seen  : {len(all_ids_seen)}")
    print(f"  Avg FPS          : {avg_fps:.1f}")
    print(f"  Output video     : {output_path}")
    print(f"  Heatmap          : {config.HEATMAP_PATH}")
    print(f"  Count plot       : {config.COUNT_PLOT}")
    print(f"  Screenshots      : screenshots/")
    print("═" * 45 + "\n")

    return {
        "frames":     frame_n,
        "unique_ids": len(all_ids_seen),
        "avg_fps":    avg_fps,
    }


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    max_frames = 300 if args.test else config.MAX_FRAMES

    if args.test:
        print("\n⚡ TEST MODE — processing first 300 frames only\n")

    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        max_frames=max_frames,
        show_trails=not args.no_trails,
    )
