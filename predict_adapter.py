# =============================================================================
# backend/predict_adapter.py
# Thin adapter that calls existing predict.py / model.py logic from the API.
# =============================================================================

import os, sys
from pathlib import Path
import numpy as np
import torch
from collections import deque
from torchvision import transforms

BACKEND = Path(__file__).parent
sys.path.insert(0, str(BACKEND))

import config
from model import load_dl_model, load_ae_model, MLAnomalyModels
from dataset import CNNFeatureExtractor, infer_anomaly_type
from utils import optical_flow_magnitude

_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_all_models() -> dict:
    """Load all models once; return as dict for reuse."""
    models = {}
    try:
        models["dl"]        = load_dl_model(config.DL_MODEL_PATH, config.DEVICE)
        models["ae"]        = load_ae_model(config.AE_MODEL_PATH, config.DEVICE)
        models["extractor"] = CNNFeatureExtractor(config.DEVICE)
        try:
            models["ml"] = MLAnomalyModels.load(config.ML_MODEL_PATH)
        except Exception:
            models["ml"] = None
    except Exception as e:
        print(f"[WARN] Model load partial: {e}")
    return models


def run_inference_frame(bgr_frame: np.ndarray,
                        clip_buf: list,
                        prev_bgr,
                        models: dict,
                        threshold: float = 0.5) -> dict:
    """
    Process a single BGR frame. Maintains clip_buf as sliding window.
    Returns dict with confidence, is_anomaly, anomaly_type.
    """
    import cv2

    # Resize and add to buffer
    rgb_sm = cv2.cvtColor(
        cv2.resize(bgr_frame, (config.FRAME_W, config.FRAME_H)),
        cv2.COLOR_BGR2RGB
    )
    clip_buf.append(rgb_sm)
    if len(clip_buf) > config.CLIP_LEN:
        clip_buf.pop(0)

    if len(clip_buf) < config.CLIP_LEN:
        return {"confidence": 0.0, "is_anomaly": False, "anomaly_type": "Normal"}

    clip_np = np.array(clip_buf, dtype=np.uint8)

    # DL inference
    dl_conf = 0.0
    if "dl" in models:
        frames_t = torch.stack([_TF(f) for f in clip_np]).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            probs = models["dl"].predict_proba(frames_t)
        dl_conf = float(probs[0, 1].cpu())

    # AE score
    ae_score = 0.0
    if "ae" in models:
        ae_in = _TF(clip_np[-1]).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            loss = models["ae"].reconstruction_loss(ae_in).item()
        ae_score = min(loss / (config.AE_THRESHOLD * 2), 1.0)

    # ML score
    ml_score = 0.5
    if models.get("ml") and models.get("extractor"):
        feat = models["extractor"].extract_clip(clip_np)
        ml_score = models["ml"].score(feat)

    # Motion
    motion = 0.0
    if prev_bgr is not None:
        motion = optical_flow_magnitude(prev_bgr, bgr_frame)

    ensemble = 0.60 * dl_conf + 0.25 * ml_score + 0.15 * ae_score
    is_anom  = ensemble >= threshold

    atype = "Normal"
    if is_anom:
        atype = infer_anomaly_type(dl_conf, ml_score, motion, 0)

    return {
        "confidence"  : round(ensemble, 4),
        "dl_conf"     : round(dl_conf, 4),
        "ml_score"    : round(ml_score, 4),
        "ae_score"    : round(ae_score, 4),
        "is_anomaly"  : is_anom,
        "anomaly_type": atype,
        "motion"      : round(motion, 2),
    }
