"""
Utility functions for the Smart Surveillance System.

This module contains helper functions for:
- Seeding randomness for reproducibility
- Loading YOLO object detector (optional)
- Drawing bounding boxes and anomaly scores on frames
- Simple logging helpers
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

from config import YOLO_CONFIDENCE_THRESHOLD, USE_YOLO


def setup_logging() -> None:
    """Configure a basic logging format for the whole project."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yolo_detector(device: torch.device):
    """
    Load a YOLOv8 detector from the ultralytics package if available.

    YOLO is used in this project to detect people and objects and then
    combine those detections with anomaly scores from deep models.
    """
    if not USE_YOLO:
        return None

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        logging.warning(
            "ultralytics package not found. Install with 'pip install ultralytics' "
            "or set USE_YOLO = False in config.py to silence this message."
        )
        return None

    # Using a small model for better real-time performance
    model = YOLO("yolov8n.pt")
    model.to(device)
    return model


def run_yolo_detection(
    yolo_model,
    frame_bgr: np.ndarray,
    classes_of_interest: Optional[List[int]] = None,
) -> List[Tuple[int, int, int, int, float, int]]:
    """
    Run YOLO detection on a single frame.

    Returns a list of detections as:
    (x1, y1, x2, y2, confidence, class_id)
    """
    if yolo_model is None:
        return []

    results = yolo_model.predict(
        source=frame_bgr,
        verbose=False,
        conf=YOLO_CONFIDENCE_THRESHOLD,
    )
    detections: List[Tuple[int, int, int, int, float, int]] = []

    if not results:
        return detections

    res = results[0]
    boxes = res.boxes
    if boxes is None:
        return detections

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        if classes_of_interest is not None and cls not in classes_of_interest:
            continue
        detections.append((x1, y1, x2, y2, conf, cls))

    return detections


def draw_bounding_boxes(
    frame_bgr: np.ndarray,
    detections: List[Tuple[int, int, int, int, float, int]],
    anomaly_score: float,
    anomaly_threshold: float,
) -> np.ndarray:
    """
    Draw bounding boxes and anomaly information on a frame.

    If the anomaly score is above the threshold, the boxes and text
    are drawn in red; otherwise in green.
    """
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    is_anomaly = anomaly_score >= anomaly_threshold
    color = (0, 0, 255) if is_anomaly else (0, 255, 0)

    # Draw detections (if YOLO is enabled)
    for (x1, y1, x2, y2, conf, cls) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"id:{cls} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # Draw anomaly score and status at the top of the frame
    status_text = f"Anomaly score: {anomaly_score:.3f}"
    if is_anomaly:
        status_text += "  [ANOMALY]"

    cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        status_text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255) if is_anomaly else (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
