// frontend/src/pages/Logs.jsx
import { useEffect, useState } from 'react'
import { logsAPI } from '../services/api'
import { RefreshCw, Trash2, Search } from 'lucide-react'
import toast from 'react-hot-toast'

export default function Logs() {
  const [logs, setLogs] = useState([])
  const [total, setTotal] = useState(0)
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const data = await logsAPI.fetch(200)
      setLogs(data.logs || [])
      setTotal(data.total || 0)
    } catch { toast.error('Failed to load logs') }
    finally { setLoading(false) }
  }

  useEffect(() => { load(); const t = setInterval(load, 8000); return () => clearInterval(t) }, [])

  const clearAll = async () => {
    await logsAPI.clear()
    setLogs([]); setTotal(0)
    toast.success('Logs cleared')
  }

  const filtered = search
    ? logs.filter(r => Object.values(r).join(' ').toLowerCase().includes(search.toLowerCase()))
    : logs

  const confColor = (c) => {
    const n = parseFloat(c)
    return n > 0.75 ? 'var(--danger)' : n > 0.5 ? 'var(--warning)' : 'var(--success)'
  }

  return (
    <div style={{ padding:28, display:'flex', flexDirection:'column', gap:20, flex:1 }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <div>
          <h2 style={{ margin:0, fontSize:22, fontWeight:800 }}>Anomaly Logs</h2>
          <p style={{ margin:'4px 0 0', color:'var(--text-muted)', fontSize:14 }}>
            {total} total records
          </p>
        </div>
        <div style={{ display:'flex', gap:10 }}>
          <div style={{ position:'relative' }}>
            <Search size={14} style={{ position:'absolute', left:10, top:'50%',
              transform:'translateY(-50%)', color:'var(--text-muted)' }}/>
            <input className="input-base" style={{ paddingLeft:32, width:220 }}
              placeholder="Search logs…" value={search}
              onChange={e=>setSearch(e.target.value)}/>
          </div>
          <button className="btn-ghost" onClick={load} disabled={loading}>
            <RefreshCw size={14} style={{ marginRight:6 }}/>{loading ? '…' : 'Refresh'}
          </button>
          <button className="btn-danger" onClick={clearAll} style={{ display:'flex', alignItems:'center', gap:6 }}>
            <Trash2 size={14}/> Clear All
          </button>
        </div>
      </div>

      <div className="card" style={{ padding:0, overflow:'hidden', flex:1 }}>
        <div style={{ overflowX:'auto', maxHeight:'calc(100vh - 240px)', overflowY:'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                {['Timestamp','Video','Frame','Anomaly Type','DL Conf','ML Score','Snapshot'].map(h => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0
                ? <tr><td colSpan="7" style={{ textAlign:'center', padding:40, color:'var(--text-muted)' }}>
                    No log entries found
                  </td></tr>
                : filtered.map((row, i) => (
                  <tr key={i} className="fade-in">
                    <td>{row.timestamp}</td>
                    <td style={{ maxWidth:160, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                      {row.video}
                    </td>
                    <td>{row.frame}</td>
                    <td>
                      <span style={{
                        padding:'2px 8px', borderRadius:12, fontSize:11, fontWeight:600,
                        background:'rgba(239,68,68,0.12)', color:'var(--danger)'
                      }}>{row.anomaly_type}</span>
                    </td>
                    <td style={{ color: confColor(row.dl_confidence) }}>
                      {parseFloat(row.dl_confidence || 0).toFixed(3)}
                    </td>
                    <td style={{ color:'var(--text-secondary)' }}>
                      {parseFloat(row.ml_score || 0).toFixed(3)}
                    </td>
                    <td style={{ fontSize:11, maxWidth:120, overflow:'hidden',
                                  textOverflow:'ellipsis', color:'var(--text-muted)' }}>
                      {row.snapshot ? row.snapshot.split('/').pop() : '—'}
                    </td>
                  </tr>
                ))
              }
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}


// ─────────────────────────────────────────────────────────────────────────────
// frontend/src/pages/Snapshots.jsx
export function Snapshots() {
  const [files, setFiles] = useState([])
  const load = async () => {
    const { files: f } = await outputsAPI.snapshots().catch(() => ({ files:[] }))
    setFiles(f || [])
  }
  useEffect(() => { load() }, [])

  return (
    <div style={{ padding:28, flex:1 }}>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:24 }}>
        <div>
          <h2 style={{ margin:0, fontSize:22, fontWeight:800 }}>Anomaly Snapshots</h2>
          <p style={{ margin:'4px 0 0', color:'var(--text-muted)', fontSize:14 }}>{files.length} saved</p>
        </div>
        <button className="btn-ghost" onClick={load}><RefreshCw size={14}/></button>
      </div>
      {files.length === 0
        ? <div style={{ textAlign:'center', padding:80, color:'var(--text-muted)' }}>
            No snapshots yet. Run detection to generate them.
          </div>
        : <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(220px,1fr))', gap:16 }}>
            {files.map((f,i) => (
              <div key={i} className="card" style={{ overflow:'hidden' }}>
                <div style={{ height:140, background:'var(--bg-surface)', position:'relative',
                              display:'flex', alignItems:'center', justifyContent:'center' }}>
                  <img src={f.url} alt={f.name} style={{ maxHeight:'100%', maxWidth:'100%', objectFit:'cover' }}
                    onError={e=>e.target.style.display='none'}/>
                  <div style={{ position:'absolute', top:8, right:8, background:'rgba(239,68,68,0.85)',
                                padding:'2px 8px', borderRadius:12, fontSize:10, fontWeight:700, color:'#fff' }}>
                    ANOMALY
                  </div>
                </div>
                <div style={{ padding:'10px 14px' }}>
                  <p style={{ margin:0, fontSize:11, fontFamily:'var(--font-mono)',
                               color:'var(--text-muted)', overflow:'hidden',
                               textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{f.name}</p>
                  <p style={{ margin:'4px 0 0', fontSize:11, color:'var(--text-muted)' }}>
                    {f.size_mb} MB
                  </p>
                </div>
              </div>
            ))}
          </div>
      }
    </div>
  )
}

import { outputsAPI } from '../services/api'
import { RefreshCw as R2 } from 'lucide-react'

// frontend/src/pages/Clips.jsx
export function Clips() {
  const [files, setFiles] = useState([])
  const load = async () => {
    const { files: f } = await outputsAPI.clips().catch(() => ({ files:[] }))
    setFiles(f || [])
  }
  useEffect(() => { load() }, [])
  return (
    <div style={{ padding:28, flex:1 }}>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:24 }}>
        <div>
          <h2 style={{ margin:0, fontSize:22, fontWeight:800 }}>Anomaly Clips</h2>
          <p style={{ margin:'4px 0 0', color:'var(--text-muted)', fontSize:14 }}>Saved anomaly segments</p>
        </div>
        <button className="btn-ghost" onClick={load}><R2 size={14}/></button>
      </div>
      {files.length === 0
        ? <div style={{ textAlign:'center', padding:80, color:'var(--text-muted)' }}>
            No clips saved yet.
          </div>
        : <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
            {files.map((f,i) => (
              <div key={i} className="card" style={{ padding:16, display:'flex',
                                                      alignItems:'center', gap:16 }}>
                <div style={{ width:48, height:48, borderRadius:10, background:'rgba(239,68,68,0.12)',
                              display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0 }}>
                  <span style={{ fontSize:20 }}>🎬</span>
                </div>
                <div style={{ flex:1 }}>
                  <p style={{ margin:0, fontSize:13, fontWeight:600 }}>{f.name}</p>
                  <p style={{ margin:'3px 0 0', fontSize:12, color:'var(--text-muted)',
                               fontFamily:'var(--font-mono)' }}>{f.size_mb} MB</p>
                </div>
                <a href={f.url} download className="btn-ghost" style={{ fontSize:12, textDecoration:'none' }}>
                  Download
                </a>
              </div>
            ))}
          </div>
      }
    </div>
  )
}

// frontend/src/pages/Alerts.jsx
export function Alerts() {
  const recentDetections = useDetectionStore(s => s.recentDetections)
  const isAnomaly = useDetectionStore(s => s.isAnomaly)
  const anomalyType = useDetectionStore(s => s.anomalyType)
  const confidence = useDetectionStore(s => s.confidence)

  return (
    <div style={{ padding:28, flex:1 }}>
      <h2 style={{ margin:'0 0 6px', fontSize:22, fontWeight:800 }}>Alert Center</h2>
      <p style={{ margin:'0 0 24px', color:'var(--text-muted)', fontSize:14 }}>Real-time anomaly alerts</p>

      {isAnomaly && (
        <div className="anomaly-glow card" style={{
          padding:24, marginBottom:20,
          border:'1px solid rgba(239,68,68,0.5)',
          background:'rgba(239,68,68,0.06)'
        }}>
          <div style={{ display:'flex', alignItems:'center', gap:12 }}>
            <span style={{ fontSize:32 }}>🚨</span>
            <div>
              <h3 style={{ margin:0, color:'var(--danger)', fontSize:20, fontWeight:800 }}>
                ANOMALY DETECTED
              </h3>
              <p style={{ margin:'4px 0 0', color:'var(--text-secondary)', fontSize:14 }}>
                {anomalyType} · Confidence {Math.round(confidence*100)}%
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="card" style={{ padding:0, overflow:'hidden' }}>
        <div style={{ padding:'14px 20px', borderBottom:'1px solid var(--border)' }}>
          <span style={{ fontWeight:600 }}>Alert History (this session)</span>
        </div>
        {recentDetections.length === 0
          ? <p style={{ textAlign:'center', padding:40, color:'var(--text-muted)', fontSize:14 }}>
              No alerts triggered yet
            </p>
          : recentDetections.map((d,i) => (
            <div key={i} style={{
              display:'flex', gap:16, alignItems:'center',
              padding:'14px 20px', borderBottom:'1px solid rgba(56,189,248,0.06)'
            }}>
              <div style={{ width:8, height:8, borderRadius:'50%', background:'var(--danger)', flexShrink:0 }}/>
              <div style={{ flex:1 }}>
                <span style={{ fontSize:13, fontWeight:600, color:'var(--danger)' }}>{d.type}</span>
                <span style={{ fontSize:12, color:'var(--text-muted)', marginLeft:12,
                               fontFamily:'var(--font-mono)' }}>Frame {d.frame}</span>
              </div>
              <div style={{ textAlign:'right' }}>
                <div style={{ fontSize:14, fontWeight:700, fontFamily:'var(--font-mono)',
                               color: d.conf > 0.7 ? 'var(--danger)' : 'var(--warning)' }}>
                  {Math.round(d.conf*100)}%
                </div>
                <div style={{ fontSize:11, color:'var(--text-muted)' }}>{d.time}</div>
              </div>
            </div>
          ))
        }
      </div>
    </div>
  )
}
