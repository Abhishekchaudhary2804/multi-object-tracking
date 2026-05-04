# =============================================================
#  visualizer.py  —  Drawing annotations on frames
# =============================================================

import cv2
import numpy as np
from typing import Optional


# ── Color palette (deterministic per ID) ─────────────────────
_PALETTE = [
    (255, 80,  80),   # red
    (80,  200, 255),  # sky blue
    (80,  255, 120),  # green
    (255, 180, 40),   # amber
    (200, 80,  255),  # purple
    (255, 120, 200),  # pink
    (40,  220, 200),  # teal
    (255, 220, 60),   # yellow
    (160, 255, 80),   # lime
    (255, 140, 60),   # orange
    (80,  130, 255),  # blue
    (255, 60,  180),  # magenta
]

def get_color(track_id: int) -> tuple:
    """Return a consistent BGR color for a given track ID."""
    return _PALETTE[int(track_id) % len(_PALETTE)]


# ── Main drawing function ─────────────────────────────────────
def draw_tracks(frame: np.ndarray,
                tracked_detections,
                trails: Optional[dict] = None,
                show_confidence: bool = False,
                box_thickness: int = 2,
                font_scale: float = 0.55) -> np.ndarray:
    """
    Draw bounding boxes, unique ID labels, and optional trails.

    Args:
        frame:               BGR numpy array
        tracked_detections:  sv.Detections with tracker_id populated
        trails:              dict {track_id: [(cx,cy), ...]} or None
        show_confidence:     whether to show detection confidence
        box_thickness:       bounding box line width
        font_scale:          text size multiplier

    Returns:
        Annotated BGR frame (modified in-place)
    """
    # Draw trails FIRST (underneath boxes)
    if trails:
        _draw_trails(frame, trails)

    # No detections this frame
    if tracked_detections is None or len(tracked_detections) == 0:
        return frame

    # Handle supervision Detections object
    try:
        xyxy_list   = tracked_detections.xyxy
        tracker_ids = tracked_detections.tracker_id
        confs       = tracked_detections.confidence
    except AttributeError:
        # DeepSORT fallback format
        return _draw_deepsort(frame, tracked_detections, trails,
                              box_thickness, font_scale)

    if tracker_ids is None:
        return frame

    for i, (xyxy, tid, conf) in enumerate(zip(xyxy_list, tracker_ids, confs)):
        x1, y1, x2, y2 = map(int, xyxy)
        color = get_color(tid)

        # ── Bounding box ──────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        # ── Label text ───────────────────────────────────────
        label = f"ID {tid}"
        if show_confidence:
            label += f"  {conf:.2f}"

        font      = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        pad   = 4
        lx1   = x1
        ly1   = max(y1 - th - pad * 2, 0)
        lx2   = x1 + tw + pad * 2
        ly2   = y1

        # Filled background for readability
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)

        # White text
        cv2.putText(
            frame, label,
            (lx1 + pad, ly2 - pad),
            font, font_scale,
            (255, 255, 255), thickness,
            cv2.LINE_AA,
        )

    return frame


def _draw_trails(frame: np.ndarray, trails: dict) -> None:
    """Draw fading movement trails for each tracked ID."""
    for tid, points in trails.items():
        if len(points) < 2:
            continue
        color = get_color(tid)
        n     = len(points)
        for i in range(1, n):
            alpha     = i / n                        # fade from dim → bright
            thickness = max(1, int(3 * alpha))
            pt_color  = tuple(int(c * alpha) for c in color)
            cv2.line(frame, points[i - 1], points[i], pt_color, thickness)


def _draw_deepsort(frame, tracks, trails, box_thickness, font_scale):
    """Draw for DeepSORT track objects."""
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid  = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        color = get_color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        label = f"ID {tid}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), font,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ── HUD overlay ───────────────────────────────────────────────
def draw_hud(frame: np.ndarray, frame_num: int,
             fps: float, active_ids: int) -> np.ndarray:
    """Draw a small heads-up display in the top-right corner."""
    h, w = frame.shape[:2]

    lines = [
        f"Frame : {frame_num:>5}",
        f"FPS   : {fps:>5.1f}",
        f"Tracks: {active_ids:>5}",
    ]

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    pad   = 8
    lh    = 18

    box_w = 140
    box_h = len(lines) * lh + pad * 2
    x0    = w - box_w - 10
    y0    = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (x0 + pad, y0 + pad + lh * (i + 1) - 4),
                    font, scale, (180, 220, 255), 1, cv2.LINE_AA)

    return frame
