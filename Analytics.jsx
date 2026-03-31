// frontend/src/pages/Analytics.jsx
import { useEffect, useState } from 'react'
import { systemAPI } from '../services/api'
import { useDetectionStore } from '../store/useStore'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid, RadarChart,
  PolarGrid, PolarAngleAxis, Radar
} from 'recharts'
import { TrendingUp, Target, Clock, Zap } from 'lucide-react'

const PALETTE = [
  '#ef4444','#f59e0b','#3b82f6','#8b5cf6',
  '#ec4899','#10b981','#06b6d4','#84cc16'
]

const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background:'var(--bg-elevated)', border:'1px solid var(--border)',
      borderRadius:8, padding:'8px 14px', fontFamily:'var(--font-mono)', fontSize:12
    }}>
      {label && <p style={{ margin:'0 0 4px', color:'var(--text-muted)' }}>{label}</p>}
      {payload.map((p,i) => (
        <p key={i} style={{ margin:'2px 0', color: p.color || 'var(--text-primary)', fontWeight:600 }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
        </p>
      ))}
    </div>
  )
}

function MetricCard({ label, value, change, icon: Icon, color = 'var(--accent)' }) {
  const pos = change >= 0
  return (
    <div className="card" style={{ padding: '20px 22px' }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start' }}>
        <div>
          <p style={{ margin:'0 0 8px', fontSize:11, color:'var(--text-muted)',
                      letterSpacing:.08em, textTransform:'uppercase' }}>{label}</p>
          <p style={{ margin:0, fontSize:28, fontWeight:800,
                      fontFamily:'var(--font-mono)', lineHeight:1 }}>{value}</p>
          {change !== undefined && (
            <p style={{ margin:'6px 0 0', fontSize:12,
                        color: pos ? 'var(--success)' : 'var(--danger)' }}>
              {pos ? '▲' : '▼'} {Math.abs(change)}% vs last session
            </p>
          )}
        </div>
        <div style={{
          width:46, height:46, borderRadius:12, flexShrink:0,
          background:`${color}18`, display:'flex', alignItems:'center', justifyContent:'center'
        }}>
          <Icon size={22} color={color}/>
        </div>
      </div>
    </div>
  )
}

export default function Analytics() {
  const [stats,   setStats]   = useState(null)
  const [sysInfo, setSysInfo] = useState(null)
  const confHistory   = useDetectionStore(s => s.confHistory)
  const anomalyCount  = useDetectionStore(s => s.anomalyCount)
  const frameCount    = useDetectionStore(s => s.frameCount)
  const recentDets    = useDetectionStore(s => s.recentDetections)

  useEffect(() => {
    const load = async () => {
      const [st, si] = await Promise.allSettled([systemAPI.stats(), systemAPI.info()])
      if (st.status === 'fulfilled') setStats(st.value)
      if (si.status === 'fulfilled') setSysInfo(si.value)
    }
    load()
    const t = setInterval(load, 20000)
    return () => clearInterval(t)
  }, [])

  // ── Build chart datasets ──────────────────────────────────────────────────
  const liveConf = confHistory.map((p, i) => ({
    frame: i,
    confidence: +(p.v * 100).toFixed(1),
    threshold: 50,
  }))

  const typeBar = stats?.by_type
    ? Object.entries(stats.by_type).map(([k, v]) => ({ type: k.replace(' ','…'), count: v }))
    : []

  const hourly = stats?.hourly || []

  const pieData = stats?.by_type
    ? Object.entries(stats.by_type).map(([k, v]) => ({ name: k, value: v }))
    : []

  // Radar data from type distribution
  const radarData = typeBar.map(d => ({ subject: d.type.split('/')[0], count: d.count, fullMark: 20 }))

  // Session confidence stats
  const confVals = confHistory.map(p => p.v)
  const avgConf  = confVals.length ? (confVals.reduce((a, b) => a + b, 0) / confVals.length) : 0
  const maxConf  = confVals.length ? Math.max(...confVals) : 0
  const detRate  = frameCount ? ((anomalyCount / frameCount) * 100) : 0

  return (
    <div style={{ padding: 28, flex: 1, display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Header */}
      <div>
        <h2 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>Analytics</h2>
        <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: 14 }}>
          Detection performance · Historical trends
        </p>
      </div>

      {/* KPI row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14 }}>
        <MetricCard label="Total Anomalies"   value={stats?.total ?? anomalyCount}
          icon={Target} color="var(--danger)" change={12}/>
        <MetricCard label="Avg Confidence"
          value={`${(+(stats?.avg_confidence ?? avgConf) * 100).toFixed(1)}%`}
          icon={TrendingUp} color="var(--warning)" change={-3}/>
        <MetricCard label="Detection Rate"
          value={`${detRate.toFixed(1)}%`}
          icon={Zap} color="var(--accent)" change={5}/>
        <MetricCard label="Peak Confidence"
          value={`${(maxConf * 100).toFixed(0)}%`}
          icon={Clock} color="#8b5cf6" change={8}/>
      </div>

      {/* Row 1 */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20 }}>
        {/* Live confidence area */}
        <div className="card" style={{ padding: 22 }}>
          <div style={{ display:'flex', justifyContent:'space-between', marginBottom: 18 }}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>Live Confidence Timeline</span>
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily:'var(--font-mono)' }}>
              {liveConf.length} samples
            </span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={liveConf}>
              <defs>
                <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#0ea5e9" stopOpacity={0.4}/>
                  <stop offset="100%" stopColor="#0ea5e9" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="dg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#ef4444" stopOpacity={0.15}/>
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(56,189,248,0.06)"/>
              <XAxis dataKey="frame" hide/>
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                tickFormatter={v => `${v}%`}/>
              <Tooltip content={<Tip/>}/>
              <Area dataKey="threshold" stroke="rgba(239,68,68,0.5)"
                fill="url(#dg)" strokeDasharray="5 5" strokeWidth={1}
                dot={false} name="Threshold"/>
              <Area dataKey="confidence" stroke="#0ea5e9" fill="url(#cg)"
                strokeWidth={2.5} dot={false} name="Confidence %"/>
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Pie */}
        <div className="card" style={{ padding: 22, display:'flex', flexDirection:'column' }}>
          <span style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Anomaly Types</span>
          {pieData.length === 0
            ? <div style={{ flex:1, display:'flex', alignItems:'center', justifyContent:'center',
                             color:'var(--text-muted)', fontSize:13 }}>
                Run detection to see distribution
              </div>
            : <>
                <ResponsiveContainer width="100%" height={160}>
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="50%" outerRadius={65}
                      dataKey="value" paddingAngle={2}>
                      {pieData.map((_, i) => (
                        <Cell key={i} fill={PALETTE[i % PALETTE.length]}/>
                      ))}
                    </Pie>
                    <Tooltip content={<Tip/>}/>
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ display:'flex', flexDirection:'column', gap:5, marginTop:8 }}>
                  {pieData.map((d, i) => (
                    <div key={i} style={{ display:'flex', justifyContent:'space-between',
                                          alignItems:'center' }}>
                      <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                        <div style={{ width:8, height:8, borderRadius:'50%', flexShrink:0,
                                      background: PALETTE[i % PALETTE.length]}}/>
                        <span style={{ fontSize:11, color:'var(--text-muted)' }}>{d.name}</span>
                      </div>
                      <span style={{ fontSize:12, fontWeight:600,
                                     fontFamily:'var(--font-mono)' }}>{d.value}</span>
                    </div>
                  ))}
                </div>
              </>
          }
        </div>
      </div>

      {/* Row 2 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20 }}>
        {/* Hourly bar */}
        <div className="card" style={{ padding: 22 }}>
          <span style={{ fontSize: 14, fontWeight: 600, display:'block', marginBottom: 18 }}>
            Hourly Event Count
          </span>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={hourly.length ? hourly : [{ hour:'--', count:0 }]}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(56,189,248,0.06)"/>
              <XAxis dataKey="hour" tick={{ fontSize:10, fill:'var(--text-muted)' }}/>
              <YAxis tick={{ fontSize:10, fill:'var(--text-muted)' }} allowDecimals={false}/>
              <Tooltip content={<Tip/>}/>
              <Bar dataKey="count" fill="var(--accent)" radius={[4,4,0,0]} name="Events"/>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Type bar */}
        <div className="card" style={{ padding: 22 }}>
          <span style={{ fontSize: 14, fontWeight: 600, display:'block', marginBottom: 18 }}>
            Top Anomaly Types
          </span>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={typeBar.slice(0,5)} layout="vertical">
              <XAxis type="number" tick={{ fontSize:10, fill:'var(--text-muted)' }} allowDecimals={false}/>
              <YAxis dataKey="type" type="category" tick={{ fontSize:10, fill:'var(--text-muted)' }} width={70}/>
              <Tooltip content={<Tip/>}/>
              <Bar dataKey="count" radius={[0,4,4,0]} name="Count">
                {typeBar.slice(0,5).map((_, i) => (
                  <Cell key={i} fill={PALETTE[i % PALETTE.length]}/>
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar */}
        <div className="card" style={{ padding: 22 }}>
          <span style={{ fontSize: 14, fontWeight: 600, display:'block', marginBottom: 8 }}>
            Threat Radar
          </span>
          <ResponsiveContainer width="100%" height={180}>
            <RadarChart data={radarData.length ? radarData : [
              { subject:'Theft',count:0 },{ subject:'Fight',count:0 },
              { subject:'Intrusion',count:0 },{ subject:'Loitering',count:0 }
            ]}>
              <PolarGrid stroke="rgba(56,189,248,0.15)"/>
              <PolarAngleAxis dataKey="subject"
                tick={{ fontSize:10, fill:'var(--text-muted)' }}/>
              <Radar dataKey="count" stroke="#ef4444" fill="#ef4444"
                fillOpacity={0.2} name="Count"/>
              <Tooltip content={<Tip/>}/>
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent detections table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding:'14px 20px', borderBottom:'1px solid var(--border)',
                      display:'flex', justifyContent:'space-between', alignItems:'center' }}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>Recent Session Detections</span>
          <span style={{ fontSize: 12, color:'var(--text-muted)',
                         fontFamily:'var(--font-mono)' }}>{recentDets.length} events</span>
        </div>
        {recentDets.length === 0
          ? <p style={{ textAlign:'center', padding:32, color:'var(--text-muted)', fontSize:13 }}>
              No detections in current session
            </p>
          : <table className="data-table">
              <thead>
                <tr>
                  {['Time','Type','Confidence','Frame'].map(h => <th key={h}>{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {recentDets.map((d, i) => (
                  <tr key={i}>
                    <td>{d.time}</td>
                    <td>
                      <span style={{ padding:'2px 8px', borderRadius:12, fontSize:11,
                                     fontWeight:600, background:'rgba(239,68,68,0.12)',
                                     color:'var(--danger)' }}>{d.type}</span>
                    </td>
                    <td style={{ color: d.conf > 0.7 ? 'var(--danger)' : 'var(--warning)',
                                 fontWeight:600 }}>
                      {(d.conf * 100).toFixed(1)}%
                    </td>
                    <td>{d.frame.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
        }
      </div>
    </div>
  )
}
