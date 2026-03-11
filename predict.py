"""
Real-time and offline inference for the Smart Surveillance System.

This script:
- Loads the trained autoencoder
- Optionally loads YOLO for object/person detection
- Processes frames from webcam, video file, or CCTV/RTSP streams
- Computes anomaly scores from reconstruction error
- Optionally uses a simple ML-based thresholding
- Draws bounding boxes and anomaly scores
- Triggers alerts when anomalies are detected
- Saves an annotated output video
"""

from __future__ import annotations

import os
import time
from typing import Union

import cv2
import numpy as np
import torch

from alert import trigger_alert
from config import (
    AUTOENCODER_ANOMALY_THRESHOLD,
    AUTOENCODER_WEIGHTS,
    DEFAULT_VIDEO_SOURCE,
    FRAME_SIZE,
    OUTPUT_VIDEOS_DIR,
)
from model import build_autoencoder
from preprocess import get_transforms
from utils import (
    draw_bounding_boxes,
    load_yolo_detector,
    run_yolo_detection,
    seed_everything,
    setup_logging,
)


def get_device() -> torch.device:
    """Return best available device for inference."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_autoencoder_for_inference(device: torch.device) -> torch.nn.Module:
    """
    Load a trained autoencoder for inference.

    If no trained weights are found, the model is created with random
    weights. This is still useful for demonstration but will not give
    meaningful anomaly scores until trained.
    """
    model = build_autoencoder(device)
    if os.path.exists(AUTOENCODER_WEIGHTS):
        state = torch.load(AUTOENCODER_WEIGHTS, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded autoencoder weights from {AUTOENCODER_WEIGHTS}")
    else:
        print("WARNING: Autoencoder weights not found. Using randomly initialized model.")
    model.eval()
    return model


def compute_frame_anomaly_score(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    device: torch.device,
) -> float:
    """
    Compute anomaly score for a single frame using reconstruction error.

    Higher reconstruction error implies more unusual / suspicious content.
    """
    transform = get_transforms()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, FRAME_SIZE[::-1])
    tensor = transform(frame_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(tensor)
        mse = torch.mean((recon - tensor) ** 2).item()

    return float(mse)


def open_video_source(source: Union[int, str]):
    """
    Open a video source which can be:
    - Webcam index (e.g. 0)
    - Path to video file
    - Network/CCTV stream URL (e.g. RTSP)
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def build_video_writer(cap, out_path: str):
    """
    Create a video writer with the same resolution and FPS as the input.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    return writer


def run_inference(
    source: Union[int, str] = DEFAULT_VIDEO_SOURCE,
    output_name: str = "output_annotated.mp4",
) -> None:
    """
    Run real-time or offline anomaly detection on a video source.
    """
    setup_logging()
    seed_everything(42)

    device = get_device()
    print("Using device:", device)

    autoencoder = load_autoencoder_for_inference(device)
    yolo_model = load_yolo_detector(device)

    cap = open_video_source(source)
    out_path = os.path.join(OUTPUT_VIDEOS_DIR, output_name)
    writer = build_video_writer(cap, out_path)

    print("Press 'q' in the video window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get anomaly score using autoencoder reconstruction error
        score = compute_frame_anomaly_score(autoencoder, frame, device)

        # Run YOLO object/person detection to provide regions of interest
        detections = run_yolo_detection(yolo_model, frame)

        # Draw detections and anomaly info
        annotated = draw_bounding_boxes(
            frame,
            detections,
            anomaly_score=score,
            anomaly_threshold=AUTOENCODER_ANOMALY_THRESHOLD,
        )

        # Show and save output
        cv2.imshow("Smart Surveillance - Anomaly Detection", annotated)
        writer.write(annotated)

        # Trigger alert if an anomaly is detected
        if score >= AUTOENCODER_ANOMALY_THRESHOLD:
            trigger_alert(score, AUTOENCODER_ANOMALY_THRESHOLD)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Saved annotated video to {out_path}")


if __name__ == "__main__":
    # Example usage:
    # - Webcam: python predict.py
    # - File:   python predict.py --source data/test_videos/example.mp4
    # (You can extend this script with argparse if desired.)
    run_inference()
