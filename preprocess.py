"""
Preprocessing utilities for the Smart Surveillance System.

This module is responsible for:
- Extracting frames from videos (training and testing)
- Resizing and normalizing frames
- Preparing PyTorch Datasets and DataLoaders
"""

from __future__ import annotations

import glob
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from config import (
    FRAME_SIZE,
    FRAMES_PER_SECOND,
    SEQUENCE_LENGTH,
    TRAIN_FRAMES_DIR,
    TEST_FRAMES_DIR,
    TRAIN_VIDEOS_DIR,
    TEST_VIDEOS_DIR,
)
from utils import ensure_dir


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frames_per_second: int = FRAMES_PER_SECOND,
) -> None:
    """
    Extract frames from a single video and save them as images.

    Frames are saved as sequentially numbered PNG files.
    """
    ensure_dir(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = frames_per_second

    frame_interval = max(int(round(fps / frames_per_second)), 1)
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
            cv2.imwrite(out_path, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()


def extract_frames_for_directory(
    videos_dir: str,
    frames_dir: str,
    frames_per_second: int = FRAMES_PER_SECOND,
) -> None:
    """
    Extract frames for all video files inside a directory.

    This helper is used both for training and test directories.
    """
    ensure_dir(frames_dir)
    video_paths = sorted(
        glob.glob(os.path.join(videos_dir, "*.mp4"))
        + glob.glob(os.path.join(videos_dir, "*.avi"))
        + glob.glob(os.path.join(videos_dir, "*.mov"))
    )

    for vid_path in video_paths:
        video_name = os.path.splitext(os.path.basename(vid_path))[0]
        out_dir = os.path.join(frames_dir, video_name)
        extract_frames_from_video(vid_path, out_dir, frames_per_second)


class FrameDataset(Dataset):
    """
    PyTorch Dataset for single-frame based anomaly detection.

    For simplicity, this dataset treats all provided frames as "normal"
    during training of the autoencoder. Anomalies are detected later
    based on reconstruction error.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Collect all frame file paths.
        self.frame_paths: List[str] = sorted(
            glob.glob(os.path.join(root_dir, "*", "*.png"))
        )

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int):
        img_path = self.frame_paths[idx]
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        return img_tensor


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequence-based models like CNN+LSTM or ConvLSTM.

    Each item is a sequence of consecutive frames.
    """

    def __init__(self, root_dir: str, sequence_length: int = SEQUENCE_LENGTH, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # Build sequences by grouping frames belonging to each video subdirectory.
        self.sequences: List[List[str]] = []
        video_dirs = sorted(
            [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
        for vdir in video_dirs:
            frames = sorted(glob.glob(os.path.join(vdir, "*.png")))
            # Slide a window over frames to create sequences
            for i in range(0, len(frames) - sequence_length + 1, sequence_length):
                seq = frames[i : i + sequence_length]
                if len(seq) == sequence_length:
                    self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        frame_paths = self.sequences[idx]
        frames: List[torch.Tensor] = []

        for p in frame_paths:
            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if self.transform:
                img_tensor = self.transform(img_rgb)
            else:
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            frames.append(img_tensor)

        # Resulting tensor has shape (T, C, H, W)
        seq_tensor = torch.stack(frames, dim=0)
        return seq_tensor


def get_transforms() -> T.Compose:
    """
    Build torchvision transforms for resizing and normalization.
    """
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize(FRAME_SIZE),
            T.ToTensor(),
        ]
    )


def create_dataloaders(
    batch_size: int,
    use_sequences: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing.

    If `use_sequences` is True, sequence datasets are used, otherwise
    simple frame datasets are used.
    """
    transform = get_transforms()

    if use_sequences:
        train_dataset = SequenceDataset(TRAIN_FRAMES_DIR, transform=transform)
        test_dataset = SequenceDataset(TEST_FRAMES_DIR, transform=transform)
    else:
        train_dataset = FrameDataset(TRAIN_FRAMES_DIR, transform=transform)
        test_dataset = FrameDataset(TEST_FRAMES_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


if __name__ == "__main__":
    # Small utility entry point to run preprocessing from the command line.
    print("Extracting frames for training videos...")
    extract_frames_for_directory(TRAIN_VIDEOS_DIR, TRAIN_FRAMES_DIR)

    print("Extracting frames for testing videos...")
    extract_frames_for_directory(TEST_VIDEOS_DIR, TEST_FRAMES_DIR)

    print("Done.")
