// frontend/src/pages/Live.jsx
import { useState, useEffect, useRef } from 'react'
import { detectionAPI, createDetectionSocket } from '../services/api'
import { useDetectionStore } from '../store/useStore'
import toast from 'react-hot-toast'
import { Wifi, Camera, Play, Square, Signal, AlertTriangle, CheckCircle } from 'lucide-react'

export default function Live() {
  const store  = useDetectionStore()
  const wsRef  = useRef(null)
  const [mode, setMode]   = useState('webcam')   // 'webcam' | 'rtsp'
  const [rtsp, setRtsp]   = useState('')
  const [camIdx, setCamIdx] = useState(0)

  useEffect(() => {
    wsRef.current = createDetectionSocket(
      data => store.setStatus(data),
      ()   => {}
    )
    return () => wsRef.current?.close()
  }, [store])

  const start = async () => {
    const source      = mode === 'webcam' ? String(camIdx) : rtsp
    const source_type = mode === 'webcam' ? 'webcam' : 'rtsp'
    if (mode === 'rtsp' && !rtsp) { toast.error('Enter RTSP URL'); return }
    try {
      store.resetSession()
      await detectionAPI.start({ source, source_type, threshold: store.threshold })
      toast.success(`${mode === 'webcam' ? 'Webcam' : 'RTSP'} feed started`)
    } catch (e) {
      toast.error(e?.response?.data?.detail || 'Failed to start stream')
    }
  }

  const stop = async () => {
    await detectionAPI.stop()
    toast('Stream stopped')
  }

  const { running, confidence, isAnomaly, anomalyType, fps, frameCount, anomalyCount } = store

  return (
    <div style={{ padding: 28, flex: 1, display:'flex', flexDirection:'column', gap: 22 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>Live Stream Detection</h2>
        <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: 14 }}>
          Real-time anomaly detection from webcam or IP camera
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.4fr', gap: 20 }}>
        {/* Config panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Mode selector */}
          <div className="card" style={{ padding: 20 }}>
            <p style={{ margin: '0 0 14px', fontSize: 12, color: 'var(--text-muted)',
                        letterSpacing: .08em, textTransform: 'uppercase' }}>Stream Source</p>
            <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
              {[
                { id: 'webcam', label: 'Webcam', icon: Camera },
                { id: 'rtsp',   label: 'RTSP / IP Cam', icon: Wifi },
              ].map(({ id, label, icon: Icon }) => (
                <button key={id} onClick={() => setMode(id)} style={{
                  flex: 1, padding: '12px 0', borderRadius: 10, cursor: 'pointer',
                  border: `1px solid ${mode === id ? 'var(--accent)' : 'var(--border)'}`,
                  background: mode === id ? 'rgba(14,165,233,0.12)' : 'var(--bg-surface)',
                  color: mode === id ? 'var(--accent)' : 'var(--text-muted)',
                  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
                  transition: 'all .15s', fontSize: 12, fontWeight: 600
                }}>
                  <Icon size={20}/>
                  {label}
                </button>
              ))}
            </div>

            {mode === 'webcam' && (
              <div>
                <label style={{ fontSize: 12, color: 'var(--text-muted)', display:'block', marginBottom: 6 }}>
                  CAMERA INDEX
                </label>
                <select value={camIdx} onChange={e => setCamIdx(+e.target.value)}
                  className="input-base">
                  <option value={0}>Camera 0 (default)</option>
                  <option value={1}>Camera 1</option>
                  <option value={2}>Camera 2</option>
                </select>
              </div>
            )}

            {mode === 'rtsp' && (
              <div>
                <label style={{ fontSize: 12, color: 'var(--text-muted)', display:'block', marginBottom: 6 }}>
                  RTSP / CCTV URL
                </label>
                <input className="input-base"
                  value={rtsp}
                  onChange={e => setRtsp(e.target.value)}
                  placeholder="rtsp://admin:password@192.168.1.100:554/stream"/>
                <p style={{ margin: '8px 0 0', fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
                  Supports RTSP, RTMP, and HTTP MJPEG streams.
                  Most IP cameras use port 554.
                </p>
              </div>
            )}
          </div>

          {/* Threshold */}
          <div className="card" style={{ padding: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Anomaly Threshold</span>
              <span style={{ fontSize: 14, fontWeight: 700,
                             fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>
                {store.threshold.toFixed(2)}
              </span>
            </div>
            <input type="range" min="0.3" max="0.9" step="0.05"
              value={store.threshold}
              onChange={e => store.setThreshold(+e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent)' }}/>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', gap: 12 }}>
            <button className="btn-primary" onClick={start}
              disabled={running} style={{ flex: 1, padding: '13px 0', fontSize: 14 }}>
              <Play size={16} style={{ marginRight: 8, verticalAlign: 'middle' }}/>
              Start Live
            </button>
            <button className="btn-danger" onClick={stop}
              disabled={!running} style={{ flex: 1, padding: '13px 0', fontSize: 14 }}>
              <Square size={16} style={{ marginRight: 8, verticalAlign: 'middle' }}/>
              Stop
            </button>
          </div>

          {/* Stream info */}
          <div className="card" style={{ padding: 16 }}>
            <p style={{ margin: '0 0 12px', fontSize: 11, color: 'var(--text-muted)',
                        letterSpacing: .08em, textTransform: 'uppercase' }}>Stream Info</p>
            {[
              { k: 'Status',  v: running ? 'LIVE' : 'IDLE', c: running ? 'var(--success)' : 'var(--text-muted)' },
              { k: 'Source',  v: store.sourceType || '—',    c: 'var(--text-secondary)' },
              { k: 'FPS',     v: fps || 0,                   c: 'var(--accent)' },
              { k: 'Frames',  v: frameCount.toLocaleString(), c: 'var(--text-secondary)' },
              { k: 'Alerts',  v: anomalyCount,                c: 'var(--danger)' },
            ].map(({ k, v, c }) => (
              <div key={k} style={{ display:'flex', justifyContent:'space-between',
                                    padding:'6px 0', borderBottom:'1px solid rgba(56,189,248,0.06)' }}>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{k}</span>
                <span style={{ fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-mono)', color: c }}>
                  {v}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Live feed panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Video placeholder */}
          <div className="card" style={{ padding: 0, overflow: 'hidden', flex: 1, minHeight: 360 }}>
            {/* Feed area */}
            <div style={{
              background: '#050810',
              minHeight: 320,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              gap: 16,
            }}>
              {/* Scanline effect on feed */}
              <div style={{
                position: 'absolute', inset: 0,
                backgroundImage: 'repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.08) 2px,rgba(0,0,0,0.08) 4px)',
                pointerEvents: 'none', zIndex: 1
              }}/>

              {/* Corner brackets (CCTV frame aesthetic) */}
              {['topleft','topright','bottomleft','bottomright'].map(pos => (
                <div key={pos} style={{
                  position: 'absolute',
                  [pos.includes('top') ? 'top' : 'bottom']: 16,
                  [pos.includes('left') ? 'left' : 'right']: 16,
                  width: 24, height: 24,
                  borderTop: pos.includes('top')    ? '2px solid rgba(14,165,233,0.5)' : 'none',
                  borderBottom: pos.includes('bottom') ? '2px solid rgba(14,165,233,0.5)' : 'none',
                  borderLeft: pos.includes('left')  ? '2px solid rgba(14,165,233,0.5)' : 'none',
                  borderRight: pos.includes('right') ? '2px solid rgba(14,165,233,0.5)' : 'none',
                  zIndex: 2,
                }}/>
              ))}

              {/* Top-left feed label */}
              <div style={{
                position: 'absolute', top: 12, left: 12, zIndex: 3,
                display: 'flex', alignItems: 'center', gap: 6,
                background: 'rgba(0,0,0,0.6)', padding: '4px 10px', borderRadius: 6
              }}>
                <span className={running ? 'pulse-dot' : ''} style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: running ? 'var(--danger)' : 'var(--text-muted)'
                }}/>
                <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)',
                               color: running ? 'var(--danger)' : 'var(--text-muted)',
                               fontWeight: 700, letterSpacing: .08em }}>
                  {running ? 'REC' : 'STANDBY'} · CAM-01
                </span>
              </div>

              {/* Center content */}
              {running
                ? (
                  <div style={{ textAlign: 'center', zIndex: 2 }}>
                    <Signal size={48} color="rgba(14,165,233,0.3)"
                      style={{ margin: '0 auto 16px', display: 'block' }}/>
                    <p style={{ margin: 0, fontSize: 14, color: 'var(--text-muted)' }}>
                      Feed active — view in system window
                    </p>
                    <p style={{ margin: '6px 0 0', fontSize: 12, color: 'var(--text-muted)',
                                 fontFamily: 'var(--font-mono)' }}>
                      Use <code style={{ color: 'var(--accent)' }}>--show</code> flag for local preview
                    </p>
                  </div>
                )
                : (
                  <div style={{ textAlign: 'center', zIndex: 2 }}>
                    <Camera size={52} color="rgba(56,189,248,0.2)"
                      style={{ margin: '0 auto 16px', display: 'block' }}/>
                    <p style={{ margin: 0, fontSize: 14, color: 'var(--text-muted)' }}>
                      No active stream
                    </p>
                    <p style={{ margin: '6px 0 0', fontSize: 12, color: 'var(--text-muted)' }}>
                      Configure source and press Start Live
                    </p>
                  </div>
                )
              }

              {/* Bottom overlay — anomaly status */}
              {running && (
                <div style={{
                  position: 'absolute', bottom: 0, left: 0, right: 0, zIndex: 3,
                  background: isAnomaly
                    ? 'rgba(185,28,28,0.85)'
                    : 'rgba(0,0,0,0.65)',
                  padding: '10px 16px',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    {isAnomaly
                      ? <AlertTriangle size={16} color="#fff"/>
                      : <CheckCircle  size={16} color="var(--success)"/>
                    }
                    <span style={{ fontSize: 13, fontWeight: 700, color: '#fff' }}>
                      {isAnomaly ? `ANOMALY: ${anomalyType}` : 'NORMAL ACTIVITY'}
                    </span>
                  </div>
                  <span style={{ fontSize: 13, fontFamily: 'var(--font-mono)',
                                 fontWeight: 700, color: '#fff' }}>
                    {(confidence * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Recent events */}
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '12px 18px', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>Live Events</span>
            </div>
            <div style={{ maxHeight: 180, overflowY: 'auto' }}>
              {store.recentDetections.length === 0
                ? <p style={{ textAlign:'center', padding:28, color:'var(--text-muted)', fontSize:13 }}>
                    Events will appear here
                  </p>
                : store.recentDetections.slice(0, 8).map((d, i) => (
                  <div key={i} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: '9px 18px', borderBottom: '1px solid rgba(56,189,248,0.06)',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span style={{ width: 7, height: 7, borderRadius: '50%',
                                     background: 'var(--danger)', flexShrink: 0, display:'block' }}/>
                      <div>
                        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--danger)' }}>{d.type}</span>
                        <span style={{ fontSize: 11, color: 'var(--text-muted)',
                                       fontFamily: 'var(--font-mono)', marginLeft: 10 }}>
                          Frame {d.frame}
                        </span>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <span style={{ fontSize: 13, fontWeight: 700,
                                     fontFamily: 'var(--font-mono)', color: 'var(--warning)' }}>
                        {(d.conf * 100).toFixed(0)}%
                      </span>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{d.time}</div>
                    </div>
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
