# =============================================================
#  config.py  —  All parameters in one place
#  Edit this file to tune the pipeline for your video
# =============================================================

# ── Paths ────────────────────────────────────────────────────
VIDEO_PATH   = "input/video.mp4"          # Your downloaded video
OUTPUT_PATH  = "output/tracked.mp4"       # Annotated output
HEATMAP_PATH = "output/heatmap.jpg"
COUNT_PLOT   = "output/count_over_time.png"
SCREENSHOT_DIR = "screenshots/"

# ── Source (REQUIRED for submission) ─────────────────────────
VIDEO_SOURCE_URL = "PASTE_YOUR_YOUTUBE_URL_HERE"

# ── Detection (YOLOv8) ───────────────────────────────────────
MODEL_SIZE   = "yolov8m.pt"   # Options: yolov8n / yolov8s / yolov8m / yolov8l / yolov8x
CONFIDENCE   = 0.30           # Lower = more detections (more false positives too)
IOU_THRESHOLD= 0.45           # Non-max suppression threshold
CLASSES      = [0]            # 0 = person (COCO). Use None to detect all classes
DEVICE       = "cpu"          # "cpu" | "cuda" | "mps" (Mac)

# ── Tracking (ByteTrack) ─────────────────────────────────────
TRACK_THRESH = 0.25           # Min confidence to start a new track
TRACK_BUFFER = 30             # Frames to keep a lost track alive (handles occlusion)
MATCH_THRESH = 0.80           # IoU threshold for track-detection matching

# ── Processing ───────────────────────────────────────────────
FRAME_SKIP   = 1              # Process every N frames (1 = every frame)
MAX_FRAMES   = None           # None = full video; set to e.g. 300 for quick test

# ── Visualization ────────────────────────────────────────────
SHOW_TRAILS     = True        # Draw movement trails behind each ID
TRAIL_LENGTH    = 30          # How many past positions to draw
SHOW_CONFIDENCE = False       # Show detection confidence on label
BOX_THICKNESS   = 2
FONT_SCALE      = 0.55
