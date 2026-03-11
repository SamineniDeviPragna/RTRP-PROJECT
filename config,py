"""
Configuration file for the Smart Surveillance System.

All important paths and hyperparameters are centralized here so that
the project is easy to tune and extend.
"""

import os


# -----------------------------
# Basic paths
# -----------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Folder where raw training / testing videos are stored
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_VIDEOS_DIR = os.path.join(DATA_DIR, "train_videos")
TEST_VIDEOS_DIR = os.path.join(DATA_DIR, "test_videos")

# Folder for extracted frames
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
TRAIN_FRAMES_DIR = os.path.join(FRAMES_DIR, "train")
TEST_FRAMES_DIR = os.path.join(FRAMES_DIR, "test")

# Folder for saved PyTorch models
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
AUTOENCODER_WEIGHTS = os.path.join(MODELS_DIR, "conv_autoencoder.pth")
CNNLSTM_WEIGHTS = os.path.join(MODELS_DIR, "cnn_lstm_anomaly.pth")

# Folder for logs, plots, and output videos
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
OUTPUT_VIDEOS_DIR = os.path.join(OUTPUTS_DIR, "videos")

os.makedirs(TRAIN_VIDEOS_DIR, exist_ok=True)
os.makedirs(TEST_VIDEOS_DIR, exist_ok=True)
os.makedirs(TRAIN_FRAMES_DIR, exist_ok=True)
os.makedirs(TEST_FRAMES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)


# -----------------------------
# Data / preprocessing settings
# -----------------------------

# Target frame size used by deep models (H, W)
FRAME_SIZE = (128, 128)

# Number of frames in a temporal window for sequence models (CNN+LSTM, ConvLSTM, etc.)
SEQUENCE_LENGTH = 16

# How many frames to sample per second from videos during preprocessing
FRAMES_PER_SECOND = 5


# -----------------------------
# Training hyperparameters
# -----------------------------

DEVICE = "cuda"  # use "cuda" if available, otherwise override to "cpu" in code

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Autoencoder-specific
AUTOENCODER_LATENT_DIM = 128

# CNN+LSTM-specific
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 1


# -----------------------------
# Anomaly detection thresholds
# -----------------------------

# Reconstruction error threshold for frame-level anomaly from autoencoder
AUTOENCODER_ANOMALY_THRESHOLD = 0.02

# Isolation Forest / One-Class SVM thresholds (used after training them)
ML_ANOMALY_SCORE_THRESHOLD = 0.5


# -----------------------------
# Video inference settings
# -----------------------------

# Default video source:
# 0 -> webcam, "path/to/video.mp4" -> file, "rtsp://..." -> CCTV/RTSP stream
DEFAULT_VIDEO_SOURCE = 0

# Whether to run YOLO-based object detection during inference (requires ultralytics)
USE_YOLO = True

# Minimum confidence for YOLO detections
YOLO_CONFIDENCE_THRESHOLD = 0.4


# -----------------------------
# Misc
# -----------------------------

RANDOM_SEED = 42
