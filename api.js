// frontend/src/services/api.js
// Centralized API layer — all backend calls go through here.

import axios from 'axios'

const BASE = '/api'

const http = axios.create({ baseURL: BASE })

// Attach auth token to every request
http.interceptors.request.use(cfg => {
  const token = localStorage.getItem('token')
  if (token) cfg.headers.Authorization = `Bearer ${token}`
  return cfg
})

// ── Auth ─────────────────────────────────────────────────────────────────────
export const authAPI = {
  login : (username, password) =>
    http.post('/auth/login', { username, password }).then(r => r.data),
  logout: () => http.post('/auth/logout').then(r => r.data),
}

// ── Upload ────────────────────────────────────────────────────────────────────
export const uploadAPI = {
  uploadVideo: (file, onProgress) => {
    const fd = new FormData()
    fd.append('file', file)
    return http.post('/upload', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: e => onProgress?.(Math.round(e.loaded*100/e.total))
    }).then(r => r.data)
  }
}

// ── Detection ─────────────────────────────────────────────────────────────────
export const detectionAPI = {
  start : (payload)  => http.post('/detection/start', payload).then(r => r.data),
  stop  : ()         => http.post('/detection/stop').then(r => r.data),
  status: ()         => http.get('/detection/status').then(r => r.data),
}

// ── Logs ──────────────────────────────────────────────────────────────────────
export const logsAPI = {
  fetch: (limit=100, skip=0) =>
    http.get(`/logs?limit=${limit}&skip=${skip}`).then(r => r.data),
  clear: () => http.delete('/logs').then(r => r.data),
}

// ── Outputs ───────────────────────────────────────────────────────────────────
export const outputsAPI = {
  videos   : () => http.get('/outputs/videos').then(r => r.data),
  snapshots: () => http.get('/outputs/snapshots').then(r => r.data),
  clips    : () => http.get('/outputs/clips').then(r => r.data),
  deleteSnap: (name) => http.delete(`/outputs/snapshots/${name}`).then(r=>r.data),
  deleteClip: (name) => http.delete(`/outputs/clips/${name}`).then(r=>r.data),
}

// ── System ────────────────────────────────────────────────────────────────────
export const systemAPI = {
  info   : () => http.get('/system/info').then(r => r.data),
  stats  : () => http.get('/stats/summary').then(r => r.data),
  health : () => http.get('/health').then(r => r.data),
}

// ── WebSocket factory ─────────────────────────────────────────────────────────
export function createDetectionSocket(onMessage, onClose) {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${window.location.host}/ws/detection`)
  ws.onmessage = e => onMessage(JSON.parse(e.data))
  ws.onclose   = () => onClose?.()
  ws.onerror   = e => console.error('WS error', e)
  return ws
}
