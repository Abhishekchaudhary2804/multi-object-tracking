# =============================================================
#  detector.py  —  YOLOv8-based person detection
# =============================================================

from ultralytics import YOLO
import numpy as np
import torch


class PersonDetector:
    """
    Wraps YOLOv8 for clean, reusable detection.

    Returns detections as numpy array: [x1, y1, x2, y2, confidence, class_id]
    """

    def __init__(self, model_path: str, confidence: float,
                 iou: float, classes: list, device: str = "cpu"):
        print(f"[Detector] Loading model: {model_path} on {device}")
        self.model      = YOLO(model_path)
        self.confidence = confidence
        self.iou        = iou
        self.classes    = classes
        self.device     = device
        print(f"[Detector] Ready ✓  (conf={confidence}, iou={iou}, classes={classes})")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run inference on a single frame.

        Args:
            frame: BGR numpy array (H, W, 3) from cv2.read()

        Returns:
            numpy array of shape (N, 6): [x1, y1, x2, y2, conf, cls]
            Returns empty array if no detections.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,   # Suppress per-frame prints
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6))

        # Stack into [x1, y1, x2, y2, conf, cls]
        data = boxes.data.cpu().numpy()  # already in xyxy format
        return data

    def get_model_info(self) -> dict:
        """Return model metadata for the report."""
        return {
            "model":      str(self.model.model),
            "task":       "detection",
            "confidence": self.confidence,
            "iou":        self.iou,
            "classes":    self.classes,
            "device":     self.device,
        }
