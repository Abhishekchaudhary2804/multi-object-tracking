# =============================================================
#  tracker.py  —  Persistent ID tracking via ByteTrack
# =============================================================

import numpy as np
from collections import defaultdict, deque
import supervision as sv


class PersistentTracker:
    def __init__(self, track_thresh=0.25, track_buffer=30, match_thresh=0.80):
        # supervision 0.22+ uses different parameter names
        try:
            # supervision >= 0.22
            self.tracker = sv.ByteTrack(
                minimum_matching_threshold=match_thresh,
                lost_track_buffer=track_buffer,
                minimum_consecutive_frames=3,
            )
        except TypeError:
            # older supervision
            self.tracker = sv.ByteTrack(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
            )
        print(f"[Tracker] ByteTrack ready (buffer={track_buffer})")

    def update(self, detections, frame_shape):
        if len(detections) == 0:
            return sv.Detections.empty()
        det = sv.Detections(
            xyxy=detections[:, :4],
            confidence=detections[:, 4],
            class_id=detections[:, 5].astype(int),
        )
        return self.tracker.update_with_detections(det)


TRACKER_BACKEND = "ByteTrack"


class TrailManager:
    def __init__(self, max_len=30):
        self.trails = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, track_id, cx, cy):
        self.trails[track_id].append((cx, cy))

    def get(self, track_id):
        return list(self.trails[track_id])

    def get_all(self):
        return dict(self.trails)