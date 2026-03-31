// frontend/src/store/useStore.js
import { create } from 'zustand'

export const useAuthStore = create((set) => ({
  user : null,
  token: localStorage.getItem('token'),
  login : (user, token) => {
    localStorage.setItem('token', token)
    localStorage.setItem('user', JSON.stringify(user))
    set({ user, token })
  },
  logout: () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    set({ user: null, token: null })
  },
}))

export const useDetectionStore = create((set, get) => ({
  // WebSocket & status
  running      : false,
  frameCount   : 0,
  anomalyCount : 0,
  confidence   : 0,
  isAnomaly    : false,
  anomalyType  : 'Normal',
  fps          : 0,
  elapsedSec   : 0,
  sourceType   : '',

  // Upload
  uploadedFile : null,
  uploadProgress: 0,

  // Config
  threshold    : 0.5,
  showYolo     : true,
  rtspUrl      : '',

  // Confidence history for chart (last 60 points)
  confHistory  : [],

  // Recent detections
  recentDetections: [],

  // Setters
  setStatus: (s) => {
    const history = [...get().confHistory, {
      t: Date.now(), v: s.confidence
    }].slice(-60)

    const recent = s.is_anomaly
      ? [{ time: new Date().toLocaleTimeString(), type: s.anomaly_type,
           conf: s.confidence, frame: s.frame_count },
         ...get().recentDetections].slice(0, 20)
      : get().recentDetections

    set({
      running      : s.running,
      frameCount   : s.frame_count,
      anomalyCount : s.anomaly_count,
      confidence   : s.confidence,
      isAnomaly    : s.is_anomaly,
      anomalyType  : s.anomaly_type,
      fps          : s.fps,
      elapsedSec   : s.elapsed_sec,
      sourceType   : s.source_type,
      confHistory  : history,
      recentDetections: recent,
    })
  },
  setUploadedFile : (f)  => set({ uploadedFile: f }),
  setUploadProgress:(p)  => set({ uploadProgress: p }),
  setThreshold    : (t)  => set({ threshold: t }),
  setRtspUrl      : (u)  => set({ rtspUrl: u }),
  resetSession    : ()   => set({
    running:false, frameCount:0, anomalyCount:0,
    confidence:0, isAnomaly:false, anomalyType:'Normal',
    fps:0, elapsedSec:0, confHistory:[], recentDetections:[]
  }),
}))
