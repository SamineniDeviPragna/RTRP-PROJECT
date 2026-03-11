# RTRP-PROJECT
**Smart Surveillance System with Anomaly Detection (PyTorch)**
This project is a beginner-friendly but realistic smart surveillance system built in Python using PyTorch.
It takes a video file, webcam feed, or CCTV/RTSP stream as input and performs real-time anomaly detection.

The system combines:

Deep Learning
Convolutional Autoencoder (frame reconstruction–based anomaly detection)
CNN + LSTM (sequence modeling; ready for future extension)
YOLO (via ultralytics) for person/object detection and region-of-interest highlighting
Machine Learning
Isolation Forest on reconstruction errors for optional anomaly scoring
Anomalous activities such as unusual movements, intrusions, fights, loitering, or abandoned objects tend to produce higher anomaly scores and will be visually highlighted.

**1. Folder Structure**
RTRP-PROJECT/
├── main.py                 # Single entry point (CLI) for preprocess/train/predict

├── config.py               # All paths, hyperparameters, thresholds

├── preprocess.py           # Frame extraction + PyTorch datasets/dataloaders

├── model.py                # Conv Autoencoder + CNN+LSTM models

├── train.py                # Training loop (autoencoder + Isolation Forest)

├── predict.py              # Real-time / offline inference + video output

├── utils.py                # Helper utilities (YOLO loader, drawing, seeding)

├── alert.py                # Alert / alarm triggering logic

├── requirements.txt        # Python dependencies

├── README.md               # This documentation

├── data/
│   ├── train_videos/       # Place training videos here

│   ├── test_videos/        # Place test videos here

│   └── frames/
│       ├── train/          # Extracted frames (auto-created)

│       └── test/           # Extracted frames (auto-created)

├── saved_models/
│   └── conv_autoencoder.pth  # Trained autoencoder weights (after training)

└── outputs/
    ├── plots/              # Training curves, CSV with reconstruction errors
    
    └── videos/             # Output videos with drawn anomalies
All folders under data/, saved_models/, and outputs/ are created automatically when you run the code.

**2. Installation**
From the root of the project (RTRP-PROJECT):

python -m venv venv
venv\Scripts\activate      # On Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
If you do not want to use YOLO, you can skip ultralytics by removing it from requirements.txt and setting USE_YOLO = False in config.py.

**3. Sample Datasets for Video Anomaly Detection**
You can use any of the following public datasets (download manually and place video files in data/train_videos and data/test_videos):

UCSD Anomaly Detection Dataset
Pedestrian walkways with anomalies such as bicycles, skateboards, etc.
Avenue Dataset
Surveillance videos with various unusual events.
Large-scale anomaly detection benchmark for surveillance.
Start by:

Putting normal videos in data/train_videos/
Putting mixture of normal + anomalous videos in data/test_videos/
The autoencoder is trained on the normal training set; anomalies are detected later using high reconstruction error.

**4. Explanation of Algorithms and Why They Are Used**
Convolutional Autoencoder (Deep Learning)

Learns to reconstruct normal frames only.
At inference time, frames with unusual patterns (fights, intrusions, abnormal motion) will have a higher reconstruction error, acting as an anomaly score.
CNN + LSTM (Deep Learning)

CNN extracts spatial features from each frame.
LSTM models temporal dynamics across a sequence of frames.
This is ideal for activities that are anomalous due to how they evolve over time (e.g., sudden running, crowd panic, fights).
In this starter project, the class definition and data pipeline are ready (model.py, SequenceDataset in preprocess.py), so you can plug it into a future training script.
YOLO (Deep Learning Object Detection)

Detects persons and objects in each frame.
Gives bounding boxes used to localize regions where anomalies may occur.
Combined with anomaly score, you can highlight suspicious persons/regions.
Isolation Forest / One-Class ML (Machine Learning)

Trained on reconstruction errors from autoencoder.
Learns a model of what error range is normal, flagging outliers as anomalies.
Useful as a classical ML layer on top of deep features, making the system more robust.

**5. Module-by-Module Explanation**
**config.py**

Central place for paths, hyperparameters, and thresholds.
Examples:
FRAME_SIZE, SEQUENCE_LENGTH, FRAMES_PER_SECOND
Training: BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
Anomaly thresholds: AUTOENCODER_ANOMALY_THRESHOLD
Video input: DEFAULT_VIDEO_SOURCE, YOLO usage flag.

**preprocess.py**

Extracts frames from videos and stores them as PNG images.
Provides:
extract_frames_from_video and extract_frames_for_directory
FrameDataset for autoencoder training (single-frame level)
SequenceDataset for sequence models (CNN+LSTM, ConvLSTM)
create_dataloaders which returns DataLoader objects for training/testing.

**model.py**

ConvAutoencoder: convolutional encoder–decoder architecture.
Used for unsupervised anomaly detection via reconstruction error.
CNNLSTMAnomalyDetector: CNN backbone + LSTM sequence model.
Learns to model activities over time and output an anomaly score.
Helper functions build_autoencoder and build_cnn_lstm.

**train.py**

Loads FrameDataset from preprocess.py.
Trains ConvAutoencoder using MSE reconstruction loss.
Saves:
Trained model weights to saved_models/conv_autoencoder.pth
Training loss curve (outputs/plots/autoencoder_loss.png)
CSV file of reconstruction errors and Isolation Forest predictions.
Fits an Isolation Forest on reconstruction errors (optional ML anomaly scorer).

**predict.py**
Handles real-time and offline inference.
Opens:
Webcam (0)
Video file path
CCTV/RTSP URL
Per frame:
Computes anomaly score using trained autoencoder.
Runs YOLO to detect and draw bounding boxes around persons/objects.
Overlays anomaly score and status on the frame.
Triggers an alert (via alert.py) if anomaly score exceeds threshold.
Saves annotated output video to outputs/videos/.

**utils.py**

setup_logging, seed_everything, ensure_dir.
load_yolo_detector and run_yolo_detection (using ultralytics.YOLO).
draw_bounding_boxes to overlay bounding boxes and anomaly text on frames.

**alert.py**

trigger_alert currently:
Prints a message to the terminal.
Tries a tiny beep placeholder.
Easy to extend with email/SMS/REST API/cloud functions.

**main.py**

Simple CLI wrapper with subcommands:
preprocess
train
predict

**6. Step-by-Step Execution (VS Code or Terminal)**
From the project root (RTRP-PROJECT), after installing dependencies:

Prepare data

Put your normal training videos into:
data/train_videos/
Put your test videos (normal + anomalous) into:
data/test_videos/
Extract frames (preprocessing)

python main.py preprocess
This fills:

data/frames/train/VIDEO_NAME/frame_000000.png, ...
data/frames/test/VIDEO_NAME/frame_000000.png, ...
Train the autoencoder + Isolation Forest
python main.py train
This will:

Train the conv autoencoder on normal frames.
Save weights to saved_models/conv_autoencoder.pth.
Save loss curve and reconstruction error stats in outputs/plots/.
Run real-time or offline inference
Webcam

python main.py predict
Video file

python main.py predict --source data/test_videos/your_video.mp4
CCTV/RTSP stream

python main.py predict --source "rtsp://user:pass@ip:port/your_stream"
During inference you will see:

Bounding boxes around detected objects (using YOLO, if enabled).
Anomaly score displayed at the top.
[ANOMALY] tag when score exceeds threshold.
Terminal alert messages when anomaly is triggered.
A saved annotated video in outputs/videos/.
Press q in the video window to stop.
**7. How to Extend the Project**
This codebase is structured to be easy to extend:

Add face recognition

Use a face detector/recognizer (e.g., face_recognition library or a deep model).
Integrate into predict.py, running after YOLO or in parallel.
Overlay recognized identities and treat unknown faces in restricted areas as anomalies.
Weapon detection

Fine-tune YOLO on weapon classes (gun, knife, etc.) or use an off-the-shelf weapon detector.
If a weapon is detected, force a high anomaly score and trigger critical alerts.
Crowd monitoring

Count people using YOLO detections.
Track crowd density and motion, raising anomalies when counts or motion patterns exceed safe thresholds.
Cloud deployment

Wrap predict.py logic in a REST API (Flask/FastAPI).
Stream video to/from a server or edge device.
Connect alert.py to cloud notifications (e.g., AWS SNS, Twilio).
Because the code is modular (config.py, preprocess.py, model.py, train.py, predict.py, alert.py, utils.py), you can safely plug in more advanced models (3D CNNs, ConvLSTMs, transformers) as your skills grow.

**8. Notes and Tips**
Real-time performance

Use a smaller input size (FRAME_SIZE) and a light YOLO model (yolov8n.pt).
Run on GPU (CUDA) when available.
Threshold tuning

Start with AUTOENCODER_ANOMALY_THRESHOLD from config.py.
Inspect the reconstruction_errors.csv and autoencoder_loss.png to tune this threshold.
Safety

This project is an educational baseline, not a production-grade security system.
Always validate and harden models before real-world deployment.
