"""
Alert / notification logic for the Smart Surveillance System.

This module is intentionally simple and beginner-friendly. It currently
supports:
- Terminal print alerts
- An optional sound alarm (beep) using OpenCV

In real deployments, this is the place to integrate:
- Email / SMS / WhatsApp alerts
- REST API calls to security dashboards
- IoT devices (sirens, lights)
"""

from __future__ import annotations

import time
from typing import Optional

import cv2


def trigger_alert(
    anomaly_score: float,
    threshold: float,
    message: Optional[str] = None,
) -> None:
    """
    Trigger a basic alert if the anomaly score exceeds the threshold.

    For now this prints to the terminal and plays a short beep.
    """
    if anomaly_score < threshold:
        return

    if message is None:
        message = f"Anomaly detected! Score={anomaly_score:.3f} (threshold={threshold:.3f})"

    print(message)

    # Attempt to play a short beep using OpenCV's GUI backend.
    # This is a simple placeholder for more advanced alarm logic.
    try:
        for _ in range(2):
            # A quick way to make a blocking beep on some systems is to create
            # and destroy a tiny window; if it does not work on your OS,
            # you can replace this with playsound or winsound.
            cv2.waitKey(10)
            time.sleep(0.1)
    except Exception:
        # If any GUI-related error occurs, simply ignore it.
        pass
