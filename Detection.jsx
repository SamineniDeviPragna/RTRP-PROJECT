// frontend/src/pages/Detection.jsx
import { useState, useEffect, useRef, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { detectionAPI, uploadAPI, createDetectionSocket } from '../services/api'
import { useDetectionStore } from '../store/useStore'
import toast from 'react-hot-toast'
import { Upload, Play, Square, Sliders, AlertTriangle,
         CheckCircle, Cpu, Zap, BarChart2, Clock } from 'lucide-react'

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusBadge({ isAnomaly, type }) {
  return (
    <span style={{
      display:'inline-flex', alignItems:'center', gap:6,
      padding:'5px 12px', borderRadius:20, fontSize:12, fontWeight:600,
      background: isAnomaly ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.12)',
      color: isAnomaly ? 'var(--danger)' : 'var(--success)',
      border: `1px solid ${isAnomaly ? 'rgba(239,68,68,0.3)' : 'rgba(34,197,94,0.25)'}`,
    }}>
      <span className={isAnomaly ? 'pulse-dot' : ''} style={{
        width:6, height:6, borderRadius:'50%',
        background: isAnomaly ? 'var(--danger)' : 'var(--success)'
      }}/>
      {isAnomaly ? type || 'ANOMALY' : 'NORMAL'}
    </span>
  )
}

function StatCard({ label, value, icon:Icon, color='var(--accent)' }) {
  return (
    <div className="card" style={{ padding:'16px 20px', display:'flex', alignItems:'center', gap:14 }}>
      <div style={{
        width:42, height:42, borderRadius:10, flexShrink:0,
        background:`${color}20`, display:'flex', alignItems:'center', justifyContent:'center'
      }}>
        <Icon size={20} color={color}/>
      </div>
      <div>
        <div style={{ fontSize:22, fontWeight:700, fontFamily:'var(--font-mono)', lineHeight:1 }}>{value}</div>
        <div style={{ fontSize:11, color:'var(--text-muted)', marginTop:3, letterSpacing:.05em }}>{label}</div>
      </div>
    </div>
  )
}

function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100)
  const col  = pct > 70 ? 'var(--danger)' : pct > 50 ? 'var(--warning)' : 'var(--success)'
  return (
    <div>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:6 }}>
        <span style={{ fontSize:12, color:'var(--text-muted)' }}>Anomaly Confidence</span>
        <span style={{ fontSize:14, fontWeight:700, fontFamily:'var(--font-mono)', color:col }}>
          {pct}%
        </span>
      </div>
      <div style={{ height:8, background:'var(--bg-surface)', borderRadius:4, overflow:'hidden' }}>
        <div className="conf-bar" style={{
          height:'100%', width:`${pct}%`, background:col,
          borderRadius:4, transition:'width 0.3s ease, background 0.3s'
        }}/>
      </div>
    </div>
  )
}

// ── Main Detection Page ───────────────────────────────────────────────────────

export default function Detection() {
  const store    = useDetectionStore()
  const wsRef    = useRef(null)
  const [tab, setTab]           = useState('upload')   // 'upload' | 'webcam' | 'rtsp'
  const [uploading, setUploading] = useState(false)
  const [uploadedMeta, setUploadedMeta] = useState(null)

  // ── WebSocket lifecycle ──────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    wsRef.current = createDetectionSocket(
      (data) => store.setStatus(data),
      ()     => { if (store.running) setTimeout(connectWS, 2000) }
    )
  }, [store])

  useEffect(() => {
    connectWS()
    return () => wsRef.current?.close()
  }, [connectWS])

  // ── File upload (drag & drop) ────────────────────────────────────────────
  const onDrop = useCallback(async (files) => {
    const file = files[0]
    if (!file) return
    setUploading(true)
    try {
      const meta = await uploadAPI.uploadVideo(file, p => store.setUploadProgress(p))
      setUploadedMeta(meta)
      store.setUploadedFile(meta.filename)
      toast.success(`Uploaded: ${meta.filename}`)
    } catch { toast.error('Upload failed') }
    finally { setUploading(false) }
  }, [store])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'video/*': ['.mp4','.avi','.mov','.mkv'] }, multiple: false
  })

  // ── Start / Stop ─────────────────────────────────────────────────────────
  const startDetection = async () => {
    let source = '', sourceType = ''
    if (tab === 'upload') {
      if (!store.uploadedFile) { toast.error('Upload a video first'); return }
      source = store.uploadedFile; sourceType = 'video'
    } else if (tab === 'webcam') {
      source = '0'; sourceType = 'webcam'
    } else {
      if (!store.rtspUrl) { toast.error('Enter RTSP URL'); return }
      source = store.rtspUrl; sourceType = 'rtsp'
    }
    try {
      store.resetSession()
      await detectionAPI.start({ source, source_type: sourceType, threshold: store.threshold })
      connectWS()
      toast.success('Detection started')
    } catch (e) { toast.error(e?.response?.data?.detail || 'Failed to start') }
  }

  const stopDetection = async () => {
    try {
      await detectionAPI.stop()
      toast('Detection stopped')
    } catch { toast.error('Failed to stop') }
  }

  const { running, confidence, isAnomaly, anomalyType, frameCount, anomalyCount, fps, elapsedSec } = store

  return (
    <div style={{ padding:28, display:'flex', flexDirection:'column', gap:24, flex:1 }}>
      {/* Header */}
      <div>
        <h2 style={{ margin:0, fontSize:22, fontWeight:800 }}>Anomaly Detection</h2>
        <p style={{ margin:'4px 0 0', color:'var(--text-muted)', fontSize:14 }}>
          Upload video · Webcam · RTSP stream
        </p>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr 1fr', gap:14 }}>
        <StatCard label="FRAMES PROCESSED" value={frameCount.toLocaleString()} icon={BarChart2}/>
        <StatCard label="ANOMALIES FOUND"  value={anomalyCount} icon={AlertTriangle} color="var(--danger)"/>
        <StatCard label="PROCESSING FPS"   value={`${fps}`} icon={Zap} color="var(--warning)"/>
        <StatCard label="ELAPSED TIME"     value={`${elapsedSec}s`} icon={Clock}/>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1.4fr 1fr', gap:20 }}>
        {/* Left — source + controls */}
        <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
          {/* Source tabs */}
          <div className="card" style={{ padding:20 }}>
            <div style={{ display:'flex', gap:4, marginBottom:20 }}>
              {['upload','webcam','rtsp'].map(t => (
                <button key={t} onClick={()=>setTab(t)} style={{
                  padding:'7px 16px', borderRadius:8, border:'none', cursor:'pointer',
                  fontSize:12, fontWeight:600, textTransform:'uppercase', letterSpacing:.06em,
                  background: tab===t ? 'var(--accent)' : 'var(--bg-surface)',
                  color: tab===t ? '#fff' : 'var(--text-muted)'
                }}>{t}</button>
              ))}
            </div>

            {tab === 'upload' && (
              <div {...getRootProps()} style={{
                border:`2px dashed ${isDragActive ? 'var(--accent)' : 'var(--border)'}`,
                borderRadius:12, padding:40, textAlign:'center', cursor:'pointer',
                background: isDragActive ? 'rgba(14,165,233,0.06)' : 'var(--bg-surface)',
                transition:'all 0.2s'
              }}>
                <input {...getInputProps()}/>
                <Upload size={32} color={isDragActive ? 'var(--accent)' : 'var(--text-muted)'}
                  style={{ margin:'0 auto 12px', display:'block' }}/>
                {uploading
                  ? <p style={{ color:'var(--accent)', margin:0 }}>Uploading {store.uploadProgress}%…</p>
                  : <p style={{ color:'var(--text-muted)', margin:0, fontSize:14 }}>
                      Drag & drop video or <span style={{ color:'var(--accent)' }}>browse</span>
                    </p>
                }
                {uploadedMeta && (
                  <div style={{ marginTop:14, padding:'10px 14px', background:'rgba(34,197,94,0.08)',
                    border:'1px solid rgba(34,197,94,0.2)', borderRadius:8, textAlign:'left' }}>
                    <div style={{ fontSize:12, color:'var(--success)', fontFamily:'var(--font-mono)' }}>
                      ✓ {uploadedMeta.filename}
                    </div>
                    <div style={{ fontSize:11, color:'var(--text-muted)', marginTop:4 }}>
                      {uploadedMeta.duration}s · {uploadedMeta.width}×{uploadedMeta.height} · {uploadedMeta.size_mb}MB
                    </div>
                  </div>
                )}
              </div>
            )}
            {tab === 'webcam' && (
              <div style={{ padding:32, textAlign:'center', background:'var(--bg-surface)', borderRadius:10 }}>
                <Cpu size={40} color="var(--accent)" style={{ margin:'0 auto 12px', display:'block' }}/>
                <p style={{ color:'var(--text-secondary)', margin:0 }}>Will use system camera (index 0)</p>
              </div>
            )}
            {tab === 'rtsp' && (
              <div>
                <label style={{ fontSize:12, color:'var(--text-muted)', letterSpacing:.05em,
                                display:'block', marginBottom:8 }}>RTSP / CCTV URL</label>
                <input className="input-base"
                  value={store.rtspUrl}
                  onChange={e=>store.setRtspUrl(e.target.value)}
                  placeholder="rtsp://192.168.1.100:554/stream"/>
              </div>
            )}
          </div>

          {/* Settings */}
          <div className="card" style={{ padding:20 }}>
            <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:16 }}>
              <Sliders size={16} color="var(--accent)"/>
              <span style={{ fontSize:13, fontWeight:600 }}>Detection Settings</span>
            </div>
            <div>
              <div style={{ display:'flex', justifyContent:'space-between', marginBottom:8 }}>
                <span style={{ fontSize:13, color:'var(--text-muted)' }}>Anomaly Threshold</span>
                <span style={{ fontFamily:'var(--font-mono)', fontSize:13, color:'var(--accent)' }}>
                  {store.threshold.toFixed(2)}
                </span>
              </div>
              <input type="range" min="0.3" max="0.9" step="0.05"
                value={store.threshold}
                onChange={e=>store.setThreshold(+e.target.value)}
                style={{ width:'100%', accentColor:'var(--accent)' }}/>
              <div style={{ display:'flex', justifyContent:'space-between', marginTop:4 }}>
                <span style={{ fontSize:10, color:'var(--text-muted)' }}>Sensitive</span>
                <span style={{ fontSize:10, color:'var(--text-muted)' }}>Strict</span>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div style={{ display:'flex', gap:12 }}>
            <button className="btn-primary" onClick={startDetection}
              disabled={running} style={{ flex:1, padding:'12px 0', fontSize:14 }}>
              <span style={{ display:'flex', alignItems:'center', justifyContent:'center', gap:8 }}>
                <Play size={16}/> Start Detection
              </span>
            </button>
            <button className="btn-danger" onClick={stopDetection}
              disabled={!running} style={{ flex:1, padding:'12px 0', fontSize:14 }}>
              <span style={{ display:'flex', alignItems:'center', justifyContent:'center', gap:8 }}>
                <Square size={16}/> Stop
              </span>
            </button>
          </div>
        </div>

        {/* Right — live status */}
        <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
          {/* Detection status card */}
          <div className={`card ${isAnomaly ? 'anomaly-glow' : ''}`}
            style={{ padding:24, borderColor: isAnomaly ? 'rgba(239,68,68,0.4)' : undefined }}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:20 }}>
              <span style={{ fontSize:12, color:'var(--text-muted)', letterSpacing:.08em }}>
                DETECTION STATUS
              </span>
              <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                <span className={running ? 'pulse-dot' : ''} style={{
                  width:8, height:8, borderRadius:'50%',
                  background: running ? 'var(--success)' : 'var(--text-muted)'
                }}/>
                <span style={{ fontSize:12, color: running ? 'var(--success)' : 'var(--text-muted)' }}>
                  {running ? 'LIVE' : 'IDLE'}
                </span>
              </div>
            </div>

            <StatusBadge isAnomaly={isAnomaly} type={anomalyType}/>

            <div style={{ margin:'24px 0' }}>
              <ConfidenceBar value={confidence}/>
            </div>

            {isAnomaly && (
              <div style={{
                padding:14, borderRadius:10,
                background:'rgba(239,68,68,0.08)',
                border:'1px solid rgba(239,68,68,0.2)'
              }}>
                <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                  <AlertTriangle size={16} color="var(--danger)"/>
                  <span style={{ fontSize:13, fontWeight:600, color:'var(--danger)' }}>
                    {anomalyType}
                  </span>
                </div>
                <p style={{ margin:'6px 0 0', fontSize:12, color:'var(--text-muted)' }}>
                  Alert dispatched · Snapshot saved · CSV logged
                </p>
              </div>
            )}
            {!isAnomaly && running && (
              <div style={{
                padding:14, borderRadius:10,
                background:'rgba(34,197,94,0.06)',
                border:'1px solid rgba(34,197,94,0.15)'
              }}>
                <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                  <CheckCircle size={16} color="var(--success)"/>
                  <span style={{ fontSize:13, color:'var(--success)' }}>Scene normal</span>
                </div>
              </div>
            )}
          </div>

          {/* Recent detections mini-list */}
          <div className="card" style={{ padding:0, overflow:'hidden', flex:1 }}>
            <div style={{ padding:'14px 18px', borderBottom:'1px solid var(--border)' }}>
              <span style={{ fontSize:13, fontWeight:600 }}>Recent Detections</span>
            </div>
            <div style={{ maxHeight:260, overflowY:'auto' }}>
              {store.recentDetections.length === 0
                ? <p style={{ textAlign:'center', color:'var(--text-muted)', fontSize:13, padding:24 }}>
                    No detections yet
                  </p>
                : store.recentDetections.map((d, i) => (
                  <div key={i} style={{
                    padding:'10px 18px', borderBottom:'1px solid rgba(56,189,248,0.06)',
                    display:'flex', justifyContent:'space-between', alignItems:'center'
                  }}>
                    <div>
                      <div style={{ fontSize:12, fontWeight:600, color:'var(--danger)' }}>{d.type}</div>
                      <div style={{ fontSize:11, color:'var(--text-muted)', fontFamily:'var(--font-mono)' }}>
                        Frame {d.frame} · {d.time}
                      </div>
                    </div>
                    <span style={{
                      fontSize:12, fontFamily:'var(--font-mono)', fontWeight:700,
                      color: d.conf > 0.7 ? 'var(--danger)' : 'var(--warning)'
                    }}>
                      {Math.round(d.conf*100)}%
                    </span>
                  </div>
                ))
              }
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
