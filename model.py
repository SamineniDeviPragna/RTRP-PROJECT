"""
Model definitions for the Smart Surveillance System.

This module contains:
- A convolutional autoencoder for frame reconstruction–based anomaly detection
- A CNN+LSTM model for sequence modeling of activities

Both models are implemented in PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import AUTOENCODER_LATENT_DIM, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS


class ConvAutoencoder(nn.Module):
    """
    Simple convolutional autoencoder.

    Why autoencoders?
    ------------------
    We train the autoencoder only on "normal" surveillance footage.
    It learns to reconstruct normal frames well. At inference time,
    frames containing unusual behavior (fights, intrusions, etc.)
    usually lead to higher reconstruction error, which we treat as
    an anomaly score.
    """

    def __init__(self, latent_dim: int = AUTOENCODER_LATENT_DIM):
        super().__init__()

        # Encoder compresses input image into a latent vector
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Latent bottleneck
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)

        # Decoder reconstructs the image from latent vector
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        z = self.encoder(x)
        z = z.view(batch_size, -1)
        z = self.fc_mu(z)
        z = self.fc_dec(z)
        z = z.view(batch_size, 256, 8, 8)
        out = self.decoder(z)
        return out


class CNNFeatureExtractor(nn.Module):
    """
    A lightweight CNN used to extract spatial features from each frame
    before feeding them into an LSTM.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNNLSTMAnomalyDetector(nn.Module):
    """
    CNN+LSTM model for sequence-based anomaly detection.

    Why CNN+LSTM?
    -------------
    - CNN extracts spatial features from each frame (what is happening).
    - LSTM models temporal dynamics across frames (how it is happening).
    Anomalies can be identified when unusual temporal patterns appear,
    such as sudden fights, frantic movements, or intrusions.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
    ):
        super().__init__()
        self.cnn = CNNFeatureExtractor(out_dim=feature_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Final classifier gives an anomaly score between 0 and 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x: tensor of shape (B, T, C, H, W)
        """
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        feats = self.cnn(x)
        feats = feats.view(b, t, -1)
        lstm_out, _ = self.lstm(feats)
        # Use the last time step output as summary representation
        last_out = lstm_out[:, -1, :]
        score = self.classifier(last_out)
        return score.squeeze(1)


def build_autoencoder(device: torch.device) -> ConvAutoencoder:
    """Helper to create and move the autoencoder to the desired device."""
    model = ConvAutoencoder()
    model.to(device)
    return model


def build_cnn_lstm(device: torch.device) -> CNNLSTMAnomalyDetector:
    """Helper to create and move the CNN+LSTM model to the desired device."""
    model = CNNLSTMAnomalyDetector()
    model.to(device)
    return model


if __name__ == "__main__":
    # Quick sanity check to verify model shapes.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = build_autoencoder(device)
    dummy = torch.randn(4, 3, 128, 128).to(device)
    out = autoencoder(dummy)
    print("Autoencoder output shape:", out.shape)

    cnn_lstm = build_cnn_lstm(device)
    dummy_seq = torch.randn(2, 16, 3, 128, 128).to(device)
    score = cnn_lstm(dummy_seq)
    print("CNN+LSTM output shape:", score.shape)
