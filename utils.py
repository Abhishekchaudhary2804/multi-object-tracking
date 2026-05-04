# =============================================================
#  utils.py  —  Helper functions
# =============================================================

import cv2
import numpy as np
import os
import time
from pathlib import Path


# ── Video info ────────────────────────────────────────────────
def get_video_info(video_path: str) -> dict:
    """Return metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    info = {
        "path":        video_path,
        "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":         cap.get(cv2.CAP_PROP_FPS),
        "total_frames":int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_s":  cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return info


def print_video_info(info: dict):
    print("\n" + "─" * 45)
    print("  VIDEO INFO")
    print("─" * 45)
    for k, v in info.items():
        if k == "duration_s":
            print(f"  {k:<16}: {v:.1f}s  ({v/60:.1f} min)")
        elif isinstance(v, float):
            print(f"  {k:<16}: {v:.2f}")
        else:
            print(f"  {k:<16}: {v}")
    print("─" * 45 + "\n")


# ── Video writer ──────────────────────────────────────────────
def make_video_writer(output_path: str, fps: float,
                      width: int, height: int) -> cv2.VideoWriter:
    """Create a VideoWriter, auto-creating output directory."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter at: {output_path}")
    return writer


# ── Heatmap ───────────────────────────────────────────────────
class HeatmapAccumulator:
    """Accumulates player foot-positions over the whole video."""

    def __init__(self, width: int, height: int):
        self.map = np.zeros((height, width), dtype=np.float32)

    def update(self, xyxy_list):
        """Add foot-point (bottom-centre) of each bounding box."""
        for xyxy in xyxy_list:
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            cy = y2                          # foot position
            cy = min(cy, self.map.shape[0] - 1)
            cx = min(cx, self.map.shape[1] - 1)
            self.map[cy, cx] += 1

    def save(self, path: str, blur_radius: int = 25):
        """Save colour-mapped heatmap image."""
        blurred = cv2.GaussianBlur(self.map, (blur_radius * 2 + 1,) * 2, 0)
        norm    = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        coloured= cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, coloured)
        print(f"[Utils] Heatmap saved → {path}")


# ── Count-over-time plot ──────────────────────────────────────
def save_count_plot(counts: list, fps: float, output_path: str):
    """Save a plot of active track count per frame."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times = [i / fps for i in range(len(counts))]
        plt.figure(figsize=(14, 4))
        plt.fill_between(times, counts, alpha=0.25, color="#40a0ff")
        plt.plot(times, counts, color="#40a0ff", linewidth=1.5)
        plt.xlabel("Time (seconds)", fontsize=11)
        plt.ylabel("Active Tracks", fontsize=11)
        plt.title("Number of Tracked Subjects Over Time", fontsize=13)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[Utils] Count plot saved → {output_path}")
    except ImportError:
        print("[Utils] matplotlib not found — skipping count plot")


# ── Progress display ──────────────────────────────────────────
class ProgressBar:
    def __init__(self, total: int):
        self.total   = total
        self.start   = time.time()
        self.current = 0

    def update(self, n: int = 1):
        self.current += n
        pct    = self.current / max(self.total, 1)
        filled = int(30 * pct)
        bar    = "█" * filled + "░" * (30 - filled)
        elapsed= time.time() - self.start
        eta    = (elapsed / max(self.current, 1)) * (self.total - self.current)
        print(f"\r  [{bar}] {pct*100:5.1f}%  "
              f"frame {self.current}/{self.total}  "
              f"ETA {eta:.0f}s  ", end="", flush=True)

    def done(self):
        elapsed = time.time() - self.start
        print(f"\r  [{'█'*30}] 100.0%  Done in {elapsed:.1f}s            ")


# ── Screenshot helper ─────────────────────────────────────────
def save_screenshot(frame: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, frame)


# ── Ensure all output directories exist ──────────────────────
def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
