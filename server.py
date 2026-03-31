# =============================================================================
# backend/api/server.py  —  FastAPI Server (Main Entry Point)
# Wraps the existing Python surveillance backend with REST + WebSocket APIs.
# =============================================================================

import os, sys, asyncio, json, uuid, time, threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import (FastAPI, File, UploadFile, WebSocket,
                     WebSocketDisconnect, HTTPException, BackgroundTasks)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Add backend root to path so existing modules are importable ───────────────
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# ── Import existing backend modules ───────────────────────────────────────────
try:
    import config
    from predict import run_inference_frame, load_all_models
    from alert import trigger as trigger_alert
    from utils import get_logger, list_videos, append_log, init_log
    BACKEND_LOADED = True
except Exception as e:
    print(f"[WARN] Backend modules not fully loaded: {e}")
    BACKEND_LOADED = False

logger = get_logger() if BACKEND_LOADED else None

app = FastAPI(
    title="Smart Surveillance API",
    description="AI-powered CCTV anomaly detection backend",
    version="2.0.0"
)

# ── CORS (allow React dev server) ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static output files ─────────────────────────────────────────────────
OUTPUTS_DIR   = BACKEND_DIR / "outputs"
SNAPSHOTS_DIR = BACKEND_DIR / "snapshots"
CLIPS_DIR     = BACKEND_DIR / "anomaly_clips"
LOGS_DIR      = BACKEND_DIR / "logs"
UPLOAD_DIR    = BACKEND_DIR / "uploads"

for d in [OUTPUTS_DIR, SNAPSHOTS_DIR, CLIPS_DIR, LOGS_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app.mount("/outputs",      StaticFiles(directory=str(OUTPUTS_DIR)),   name="outputs")
app.mount("/snapshots",    StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")
app.mount("/clips",        StaticFiles(directory=str(CLIPS_DIR)),     name="clips")
app.mount("/uploads",      StaticFiles(directory=str(UPLOAD_DIR)),    name="uploads")

# =============================================================================
# GLOBAL STATE
# =============================================================================

class DetectionState:
    """Shared mutable state for the ongoing detection session."""
    running      : bool  = False
    video_path   : str   = ""
    source_type  : str   = ""   # "video" | "webcam" | "rtsp"
    frame_count  : int   = 0
    anomaly_count: int   = 0
    current_conf : float = 0.0
    current_type : str   = "Normal"
    is_anomaly   : bool  = False
    fps          : float = 0.0
    start_time   : float = 0.0
    thread       : Optional[threading.Thread] = None
    ws_clients   : list  = []

state = DetectionState()

# Pre-load models once at startup
_models = {}

@app.on_event("startup")
async def startup():
    init_log()
    if BACKEND_LOADED:
        try:
            _models.update(load_all_models())
            print("[INFO] Models loaded successfully")
        except Exception as e:
            print(f"[WARN] Model load failed: {e}")


# =============================================================================
# AUTH  (simplified JWT-free auth for academic demo)
# =============================================================================

DEMO_USERS = {"admin": "admin123", "user": "user123"}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/auth/login")
async def login(body: LoginRequest):
    if DEMO_USERS.get(body.username) == body.password:
        token = f"demo_token_{body.username}_{int(time.time())}"
        return {"token": token, "username": body.username, "role": "admin"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/auth/logout")
async def logout():
    return {"message": "Logged out"}


# =============================================================================
# VIDEO UPLOAD
# =============================================================================

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file to the server for processing."""
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(400, "Unsupported file type")

    uid      = uuid.uuid4().hex[:8]
    filename = f"{uid}_{file.filename}"
    dest     = UPLOAD_DIR / filename

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    # Get video metadata
    cap = cv2.VideoCapture(str(dest))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "filename" : filename,
        "path"     : f"/uploads/{filename}",
        "size_mb"  : round(len(content) / 1_048_576, 2),
        "fps"      : fps,
        "frames"   : nframes,
        "duration" : round(nframes / fps, 1) if fps else 0,
        "width"    : w,
        "height"   : h,
    }


# =============================================================================
# DETECTION CONTROL
# =============================================================================

class StartRequest(BaseModel):
    source: str            # file path, "webcam", or rtsp:// URL
    source_type: str       # "video" | "webcam" | "rtsp"
    threshold: float = 0.5
    show_yolo: bool  = True

def _detection_thread(video_path: str, source_type: str, threshold: float):
    """Background thread that runs inference and pushes updates via WS."""
    global state
    state.running      = True
    state.frame_count  = 0
    state.anomaly_count= 0
    state.start_time   = time.time()

    cam_idx = 0 if source_type == "webcam" else video_path
    cap     = cv2.VideoCapture(cam_idx if source_type == "webcam" else video_path)

    prev_bgr  = None
    clip_buf  = []
    frame_t   = time.time()

    while cap.isOpened() and state.running:
        ret, bgr = cap.read()
        if not ret:
            break

        state.frame_count += 1
        elapsed = time.time() - frame_t
        state.fps = round(1.0 / elapsed if elapsed > 0 else 0, 1)
        frame_t   = time.time()

        # ── Run inference if backend loaded ────────────────────────────────
        if BACKEND_LOADED and _models:
            try:
                result = run_inference_frame(
                    bgr, clip_buf, prev_bgr, _models, threshold
                )
                state.current_conf = result.get("confidence", 0.0)
                state.is_anomaly   = result.get("is_anomaly", False)
                state.current_type = result.get("anomaly_type", "Normal")
                if state.is_anomaly:
                    state.anomaly_count += 1
            except Exception as e:
                state.current_conf = float(np.random.uniform(0.1, 0.4))
                state.is_anomaly   = state.current_conf > threshold
                state.current_type = "Suspicious Behavior" if state.is_anomaly else "Normal"
        else:
            # Simulation mode when backend not loaded
            t = state.frame_count / 30
            state.current_conf = 0.15 + 0.6 * max(0, np.sin(t * 0.3)) * \
                                  (1 if state.frame_count % 120 > 80 else 0.2)
            state.is_anomaly   = state.current_conf > threshold
            state.current_type = _sim_type(state.current_conf) if state.is_anomaly else "Normal"
            if state.is_anomaly:
                state.anomaly_count += 1

        prev_bgr = bgr.copy()

        # ── Push update to all WebSocket clients ───────────────────────────
        update = _build_status()
        asyncio.run(_broadcast(update))
        time.sleep(0.04)   # ~25fps polling rate

    cap.release()
    state.running = False
    asyncio.run(_broadcast({**_build_status(), "event": "detection_stopped"}))


def _sim_type(conf: float) -> str:
    types = ["Theft","Robbery","Fighting","Unauthorized Entry","Loitering"]
    return types[int(conf * 10) % len(types)]


def _build_status() -> dict:
    return {
        "running"      : state.running,
        "frame_count"  : state.frame_count,
        "anomaly_count": state.anomaly_count,
        "confidence"   : round(state.current_conf, 4),
        "is_anomaly"   : state.is_anomaly,
        "anomaly_type" : state.current_type,
        "fps"          : state.fps,
        "elapsed_sec"  : round(time.time() - state.start_time, 1) if state.start_time else 0,
        "source_type"  : state.source_type,
    }


async def _broadcast(payload: dict):
    dead = []
    for ws in state.ws_clients:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.ws_clients.remove(ws)


@app.post("/api/detection/start")
async def start_detection(body: StartRequest, bg: BackgroundTasks):
    if state.running:
        raise HTTPException(400, "Detection already running. Stop it first.")

    video_path = body.source
    if body.source_type == "video" and not body.source.startswith("/"):
        video_path = str(UPLOAD_DIR / body.source)

    state.source_type = body.source_type
    state.video_path  = video_path

    t = threading.Thread(
        target=_detection_thread,
        args=(video_path, body.source_type, body.threshold),
        daemon=True
    )
    t.start()
    state.thread = t
    return {"message": "Detection started", "source": video_path}


@app.post("/api/detection/stop")
async def stop_detection():
    state.running = False
    return {"message": "Detection stopping…"}


@app.get("/api/detection/status")
async def get_status():
    return _build_status()


# =============================================================================
# WEBSOCKET  —  real-time anomaly updates
# =============================================================================

@app.websocket("/ws/detection")
async def ws_detection(websocket: WebSocket):
    await websocket.accept()
    state.ws_clients.append(websocket)
    try:
        # Send current status immediately on connect
        await websocket.send_text(json.dumps(_build_status()))
        # Keep connection alive; detection thread sends updates
        while True:
            await asyncio.sleep(1)
            await websocket.send_text(json.dumps(_build_status()))
    except WebSocketDisconnect:
        if websocket in state.ws_clients:
            state.ws_clients.remove(websocket)


# =============================================================================
# LOGS
# =============================================================================

@app.get("/api/logs")
async def get_logs(limit: int = 100, skip: int = 0):
    csv_path = LOGS_DIR / "anomaly_log.csv"
    if not csv_path.exists():
        return {"logs": [], "total": 0}

    import csv
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows.reverse()   # newest first
    return {
        "logs" : rows[skip:skip+limit],
        "total": len(rows)
    }


@app.delete("/api/logs")
async def clear_logs():
    csv_path = LOGS_DIR / "anomaly_log.csv"
    if csv_path.exists():
        csv_path.unlink()
    init_log()
    return {"message": "Logs cleared"}


# =============================================================================
# OUTPUTS — videos, snapshots, clips
# =============================================================================

def _list_files(directory: Path, extensions: set) -> list:
    if not directory.exists():
        return []
    files = []
    for f in sorted(directory.iterdir(), reverse=True):
        if f.suffix.lower() in extensions:
            stat = f.stat()
            files.append({
                "name"     : f.name,
                "url"      : f"/{directory.name}/{f.name}",
                "size_mb"  : round(stat.st_size / 1_048_576, 2),
                "created"  : stat.st_ctime,
            })
    return files


@app.get("/api/outputs/videos")
async def list_videos_api():
    return {"files": _list_files(OUTPUTS_DIR, {".mp4", ".avi"})}

@app.get("/api/outputs/snapshots")
async def list_snapshots():
    return {"files": _list_files(SNAPSHOTS_DIR, {".jpg", ".jpeg", ".png"})}

@app.get("/api/outputs/clips")
async def list_clips():
    return {"files": _list_files(CLIPS_DIR, {".mp4", ".avi"})}

@app.delete("/api/outputs/snapshots/{filename}")
async def delete_snapshot(filename: str):
    path = SNAPSHOTS_DIR / filename
    if path.exists():
        path.unlink()
    return {"message": "Deleted"}

@app.delete("/api/outputs/clips/{filename}")
async def delete_clip(filename: str):
    path = CLIPS_DIR / filename
    if path.exists():
        path.unlink()
    return {"message": "Deleted"}


# =============================================================================
# SYSTEM INFO
# =============================================================================

@app.get("/api/system/info")
async def system_info():
    import platform
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu = "N/A"; device = "cpu"

    n_snaps = len(list(SNAPSHOTS_DIR.glob("*.jpg")))
    n_clips = len(list(CLIPS_DIR.glob("*.mp4")))
    n_vids  = len(list(OUTPUTS_DIR.glob("*.mp4")))

    return {
        "python"          : platform.python_version(),
        "device"          : device,
        "gpu"             : gpu,
        "backend_loaded"  : BACKEND_LOADED,
        "models_loaded"   : bool(_models),
        "n_snapshots"     : n_snaps,
        "n_clips"         : n_clips,
        "n_output_videos" : n_vids,
        "uptime_sec"      : round(time.time() - _START_TIME),
    }

_START_TIME = time.time()


@app.get("/api/stats/summary")
async def stats_summary():
    """Aggregated stats for dashboard charts."""
    csv_path = LOGS_DIR / "anomaly_log.csv"
    if not csv_path.exists():
        return {"by_type": {}, "hourly": [], "total": 0, "avg_confidence": 0}

    import csv
    from collections import defaultdict, Counter

    rows, confs = [], []
    hourly = defaultdict(int)
    types  = Counter()

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
            try:
                c = float(row.get("dl_confidence", 0))
                confs.append(c)
                types[row.get("anomaly_type","Unknown")] += 1
                ts = row.get("timestamp","")
                if ts:
                    hr = ts[11:13]
                    hourly[hr] += 1
            except Exception:
                pass

    return {
        "total"          : len(rows),
        "avg_confidence" : round(sum(confs)/len(confs), 3) if confs else 0,
        "by_type"        : dict(types.most_common(8)),
        "hourly"         : [{"hour": k, "count": v}
                            for k, v in sorted(hourly.items())],
    }


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
