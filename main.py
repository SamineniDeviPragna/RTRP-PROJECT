"""
Entry point for the Smart Surveillance System.

This script provides a simple CLI-style interface that delegates to:
- preprocess.py     -> frame extraction
- train.py          -> model training
- predict.py        -> real-time / offline inference
"""

from __future__ import annotations

import argparse

from preprocess import (
    extract_frames_for_directory,
    TRAIN_VIDEOS_DIR,
    TRAIN_FRAMES_DIR,
    TEST_VIDEOS_DIR,
    TEST_FRAMES_DIR,
)
from train import main as train_main
from predict import run_inference


def build_arg_parser() -> argparse.ArgumentParser:
    """Create a simple argument parser."""
    parser = argparse.ArgumentParser(description="Smart Surveillance System with Anomaly Detection")

    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # Preprocess command
    subparsers.add_parser("preprocess", help="Extract frames from training and testing videos")

    # Train command
    subparsers.add_parser("train", help="Train the anomaly detection models")

    # Predict / infer command
    predict_parser = subparsers.add_parser("predict", help="Run real-time or offline inference")
    predict_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: 0 for webcam, path to video file, or CCTV/RTSP URL",
    )

    return parser


def main() -> None:
    """Parse arguments and call the right component."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        print("Extracting frames for training videos...")
        extract_frames_for_directory(TRAIN_VIDEOS_DIR, TRAIN_FRAMES_DIR)
        print("Extracting frames for testing videos...")
        extract_frames_for_directory(TEST_VIDEOS_DIR, TEST_FRAMES_DIR)
        print("Preprocessing complete.")

    elif args.command == "train":
        train_main()

    elif args.command == "predict":
        source = 0 if args.source is None else args.source
        run_inference(source=source)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
