# Smart Surveillance — Full-Stack AI Dashboard

A complete full-stack implementation connecting the Python CNN+BiLSTM anomaly detection backend with a React dashboard. Real-time detection status is pushed via WebSocket.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  React Frontend  (Vite · Zustand · Recharts)             │
│  localhost:3000                                           │
│                                                           │
│  Login → Dashboard → Detection → Live → Logs → Analytics │
└────────────────────┬─────────────────────────────────────┘
                     │ REST  +  WebSocket
┌────────────────────▼─────────────────────────────────────┐
│  FastAPI Backend  (uvicorn)                               │
│  localhost:8000                                           │
│                                                           │
│  /api/upload        /api/detection/start|stop|status      │
│  /api/logs          /api/outputs/*                        │
│  /ws/detection      (WebSocket push)                      │
└────────────────────┬─────────────────────────────────────┘
                     │ Python imports
┌────────────────────▼─────────────────────────────────────┐
│  Existing Python Modules                                  │
│                                                           │
│  predict.py · model.py · alert.py · feature_extraction.py│
│  data_preprocessing.py · config.py · utils.py            │
└──────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
project/
├── backend/
│   ├── api/
│   │   └── server.py          ← FastAPI application
│   ├── predict_adapter.py     ← Bridges API ↔ existing predict.py
│   ├── config.py              ← (existing)
│   ├── model.py               ← (existing)
│   ├── predict.py             ← (existing)
│   ├── alert.py               ← (existing)
│   ├── feature_extraction.py  ← (existing)
│   ├── data_preprocessing.py  ← (existing)
│   ├── utils.py               ← (existing)
│   ├── dataset/
│   │   ├── normal/
│   │   └── anomaly/
│   ├── models/                ← .pth model weights
│   ├── outputs/               ← annotated output videos (served as static)
│   ├── snapshots/             ← anomaly screenshots
│   ├── anomaly_clips/         ← trimmed anomaly clips
│   ├── uploads/               ← uploaded video files
│   ├── logs/                  ← anomaly_log.csv, system.log
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Login.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Detection.jsx
│   │   │   ├── Live.jsx
│   │   │   ├── Analytics.jsx
│   │   │   ├── Logs.jsx
│   │   │   └── OtherPages.jsx  (Snapshots, Clips, Alerts)
│   │   ├── components/
│   │   │   └── dashboard/
│   │   │       └── Sidebar.jsx
│   │   ├── services/
│   │   │   └── api.js          ← All HTTP + WS calls
│   │   ├── store/
│   │   │   └── useStore.js     ← Zustand global state
│   │   ├── styles/
│   │   │   └── globals.css
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
│
└── README.md
```

---

## Installation

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install FastAPI and all dependencies
pip install -r requirements.txt

# (Optional) Train models first
python main.py train
```

### Frontend

```bash
cd frontend

# Install Node.js 18+ from nodejs.org first, then:
npm install
```

---

## Running the Project

### Terminal 1 — Backend

```bash
cd backend
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: Models loaded successfully
```

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

You should see:
```
VITE v5.x  ready in 800ms
➜  Local:   http://localhost:3000/
```

### Open Browser

Navigate to `http://localhost:3000`

Login with: `admin / admin123`

---

## API Reference

### Authentication

```bash
# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
# Returns: {"token":"demo_token_admin_...","username":"admin"}
```

### Upload Video

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@dataset/anomaly/robbery.mp4"
# Returns: {"filename":"abc123_robbery.mp4","fps":25,"frames":1250,...}
```

### Start Detection

```bash
# On uploaded video
curl -X POST http://localhost:8000/api/detection/start \
  -H "Content-Type: application/json" \
  -d '{"source":"abc123_robbery.mp4","source_type":"video","threshold":0.5}'

# Webcam
curl -X POST http://localhost:8000/api/detection/start \
  -H "Content-Type: application/json" \
  -d '{"source":"0","source_type":"webcam","threshold":0.5}'

# RTSP camera
curl -X POST http://localhost:8000/api/detection/start \
  -H "Content-Type: application/json" \
  -d '{"source":"rtsp://192.168.1.100:554/stream","source_type":"rtsp","threshold":0.5}'
```

### Stop Detection

```bash
curl -X POST http://localhost:8000/api/detection/stop
```

### Poll Status

```bash
curl http://localhost:8000/api/detection/status
# Returns:
# {
#   "running": true,
#   "frame_count": 312,
#   "anomaly_count": 3,
#   "confidence": 0.8320,
#   "is_anomaly": true,
#   "anomaly_type": "Theft",
#   "fps": 24.1,
#   "elapsed_sec": 12.5
# }
```

### WebSocket (real-time updates)

```javascript
// JavaScript
const ws = new WebSocket('ws://localhost:8000/ws/detection')
ws.onmessage = (e) => {
  const data = JSON.parse(e.data)
  console.log(data.is_anomaly, data.confidence, data.anomaly_type)
}
```

### Fetch Logs

```bash
curl "http://localhost:8000/api/logs?limit=50&skip=0"
```

### List Snapshots

```bash
curl http://localhost:8000/api/outputs/snapshots
```

### Analytics Summary

```bash
curl http://localhost:8000/api/stats/summary
# Returns by_type counts, hourly distribution, avg_confidence
```

---

## How Integration Works

### 1. Frontend calls backend API

```
Detection.jsx
  → detectionAPI.start(payload)
  → POST /api/detection/start
  → server.py spawns _detection_thread()
  → thread calls run_inference_frame() from predict_adapter.py
  → predict_adapter imports predict.py, model.py, feature_extraction.py
```

### 2. Real-time updates via WebSocket

```
_detection_thread (background thread)
  → asyncio.run(_broadcast(update))   ← every frame
  → WebSocket message to all clients
  → ws.onmessage in React
  → store.setStatus(data)             ← Zustand store update
  → React re-renders status badge, confidence bar, charts
```

### 3. Alerts flow

```
predict_adapter detects anomaly
  → alert.py trigger() is called
  → CSV log entry written
  → Sound + popup (if not in cooldown)
  → WebSocket also broadcasts {is_anomaly: true, anomaly_type: "Theft"}
  → Frontend toast notification appears
  → Alert panel updates
```

### 4. Evidence files served

```
backend writes:
  outputs/output_video_timestamp.mp4
  snapshots/Theft_timestamp.jpg
  anomaly_clips/clip_timestamp.mp4

FastAPI serves them as static files:
  GET /snapshots/Theft_timestamp.jpg
  GET /clips/clip_timestamp.mp4

Frontend fetches list and renders gallery:
  GET /api/outputs/snapshots
  → [{name, url, size_mb}, ...]
  → <img src={file.url}/>
```

---

## Environment Variables

Create `backend/.env`:

```env
# Backend
HOST=0.0.0.0
PORT=8000
CORS_ORIGIN=http://localhost:3000

# Email alerts (optional)
SEND_EMAIL=false
EMAIL_FROM=your@gmail.com
EMAIL_PASS=your_app_password
EMAIL_TO=security@example.com
```

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
VITE_APP_TITLE=SurveillanceAI
```

---

## Production Deployment

```bash
# Build frontend
cd frontend && npm run build

# Serve built frontend from FastAPI
# In server.py, add after static mounts:
app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="spa")

# Run production server
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| AI/ML | PyTorch, ResNet-50, BiLSTM | Anomaly detection |
| Object Detection | YOLOv8n | Person bounding boxes |
| ML Layer | scikit-learn (IsoForest, OCSVM, RF) | Ensemble scoring |
| Video | OpenCV, Farneback Optical Flow | Frame processing |
| API | FastAPI + uvicorn | REST + WebSocket server |
| Frontend | React 18 + Vite | Dashboard UI |
| State | Zustand | Global detection state |
| Charts | Recharts | Analytics visualisation |
| Styling | CSS Variables + Tailwind | Dark surveillance theme |
| Fonts | Syne + JetBrains Mono | UI typography |

---

## Dashboard Pages

| Page | Route | What it shows |
|------|-------|---------------|
| Dashboard | `/dashboard` | KPI cards, confidence stream, type distribution, system health |
| Detection | `/dashboard/detection` | Upload + webcam + RTSP tabs, threshold slider, live status |
| Live/RTSP | `/dashboard/live` | Dedicated live stream view with CCTV-style frame |
| Alerts | `/dashboard/alerts` | Active anomaly banner + session alert history |
| Logs | `/dashboard/logs` | Searchable CSV log table with confidence colouring |
| Snapshots | `/dashboard/snapshots` | Photo grid of saved anomaly frames |
| Clips | `/dashboard/clips` | List of downloadable anomaly video clips |
| Analytics | `/dashboard/analytics` | Deep charts: confidence timeline, radar, hourly bar, top types |
