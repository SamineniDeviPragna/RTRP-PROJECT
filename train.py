"""
Training script for the Smart Surveillance System.

This module covers:
- Training a convolutional autoencoder on normal frames
- Optionally fitting a classical ML model (Isolation Forest) on
  reconstruction errors as an additional anomaly scorer
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader

from config import (
    AUTOENCODER_ANOMALY_THRESHOLD,
    AUTOENCODER_WEIGHTS,
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_EPOCHS,
    PLOTS_DIR,
    RANDOM_SEED,
)
from model import build_autoencoder
from preprocess import create_dataloaders
from utils import seed_everything, setup_logging


def get_device() -> torch.device:
    """
    Select and return the torch device. If CUDA is not available,
    this gracefully falls back to CPU.
    """
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU instead.")
        return torch.device("cpu")
    return torch.device(DEVICE)


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> List[float]:
    """
    Train the convolutional autoencoder on normal frames.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    epoch_losses: List[float] = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            imgs = batch.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}]  Loss: {epoch_loss:.6f}")

    return epoch_losses


def plot_training_curve(losses: List[float], out_path: str) -> None:
    """
    Save a plot of the training loss curve to disk.
    """
    plt.figure()
    plt.plot(losses, marker="o")
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_reconstruction_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Compute reconstruction errors for frames using a trained autoencoder.
    """
    model.eval()
    errors: List[float] = []
    criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for batch in loader:
            imgs = batch.to(device)
            recon = model(imgs)
            # Compute per-frame average pixel-wise error
            batch_errors = criterion(recon, imgs)
            batch_errors = batch_errors.view(batch_errors.size(0), -1).mean(dim=1)
            errors.extend(batch_errors.cpu().numpy().tolist())

    return np.array(errors)


def fit_isolation_forest(errors: np.ndarray) -> IsolationForest:
    """
    Fit an Isolation Forest model on reconstruction errors.

    Why Isolation Forest?
    ---------------------
    Classical machine learning anomaly detectors such as Isolation Forest
    and One-Class SVM are well-suited for modeling a 1D or low-dimensional
    feature like reconstruction error. They build an unsupervised model of
    "normal" scores and flag outliers as anomalies.
    """
    errors = errors.reshape(-1, 1)
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=RANDOM_SEED,
    )
    iso.fit(errors)
    return iso


def main() -> None:
    """Main entry point used to train models."""
    setup_logging()
    seed_everything(RANDOM_SEED)

    device = get_device()
    print("Using device:", device)

    # Create dataloaders based on single frames
    train_loader, test_loader = create_dataloaders(batch_size=BATCH_SIZE, use_sequences=False)

    # Build and train autoencoder
    autoencoder = build_autoencoder(device)
    losses = train_autoencoder(autoencoder, train_loader, device)

    # Save trained autoencoder weights
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(autoencoder.state_dict(), AUTOENCODER_WEIGHTS)
    print(f"Saved autoencoder weights to {AUTOENCODER_WEIGHTS}")

    # Plot training curve
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, "autoencoder_loss.png")
    plot_training_curve(losses, plot_path)
    print(f"Saved training loss plot to {plot_path}")

    # Compute reconstruction errors for train set and fit Isolation Forest
    errors = compute_reconstruction_errors(autoencoder, train_loader, device)
    iso = fit_isolation_forest(errors)

    # Save per-frame errors and anomaly predictions from Isolation Forest
    preds = iso.predict(errors.reshape(-1, 1))
    anomaly_flags = (preds == -1).astype(int)
    df = pd.DataFrame(
        {
            "reconstruction_error": errors,
            "isolation_forest_anomaly": anomaly_flags,
        }
    )
    csv_path = os.path.join(PLOTS_DIR, "reconstruction_errors.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved reconstruction error statistics to {csv_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
