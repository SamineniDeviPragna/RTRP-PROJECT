// frontend/src/pages/Dashboard.jsx
import { useEffect, useState } from 'react'
import { systemAPI, outputsAPI } from '../services/api'
import { useDetectionStore } from '../store/useStore'
import {
  LineChart, Line, AreaChart, Area, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar
} from 'recharts'
import { Shield, AlertTriangle, Cpu, HardDrive,
         Video, Image, Film, Activity, Zap } from 'lucide-react'

const COLORS = ['#ef4444','#f59e0b','#3b82f6','#8b5cf6','#ec4899','#10b981','#06b6d4','#84cc16']

function SysCard({ label, value, sub, icon:Icon, color='var(--accent)', ok=true }) {
  return (
    <div className="card" style={{ padding:'18px 20px' }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start' }}>
        <div>
          <p style={{ margin:'0 0 6px', fontSize:11, color:'var(--text-muted)',
                      letterSpacing:.08em, textTransform:'uppercase' }}>{label}</p>
          <p style={{ margin:0, fontSize:24, fontWeight:800, fontFamily:'var(--font-mono)',
                      color: ok ? 'var(--text-primary)' : 'var(--danger)' }}>{value}</p>
          {sub && <p style={{ margin:'4px 0 0', fontSize:12, color:'var(--text-muted)' }}>{sub}</p>}
        </div>
        <div style={{
          width:42, height:42, borderRadius:10,
          background:`${color}18`, display:'flex', alignItems:'center', justifyContent:'center'
        }}>
          <Icon size={20} color={color}/>
        </div>
      </div>
    </div>
  )
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background:'var(--bg-card)', border:'1px solid var(--border)',
                  borderRadius:8, padding:'8px 12px' }}>
      <p style={{ margin:0, fontSize:11, color:'var(--text-muted)' }}>{label}</p>
      {payload.map((p,i) => (
        <p key={i} style={{ margin:'3px 0 0', fontSize:13, fontWeight:600, color:p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed ? p.value.toFixed(3) : p.value : p.value}
        </p>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const [sysInfo, setSysInfo] = useState(null)
  const [stats,   setStats]   = useState(null)
  const [counts,  setCounts]  = useState({ snaps:0, clips:0, videos:0 })
  const confHistory = useDetectionStore(s => s.confHistory)
  const isAnomaly   = useDetectionStore(s => s.isAnomaly)
  const running     = useDetectionStore(s => s.running)
  const anomalyCount= useDetectionStore(s => s.anomalyCount)

  useEffect(() => {
    const load = async () => {
      try {
        const [si, st, sn, cl, vi] = await Promise.allSettled([
          systemAPI.info(), systemAPI.stats(),
          outputsAPI.snapshots(), outputsAPI.clips(), outputsAPI.videos()
        ])
        if (si.status==='fulfilled') setSysInfo(si.value)
        if (st.status==='fulfilled') setStats(st.value)
        setCounts({
          snaps : sn.status==='fulfilled' ? sn.value.files?.length : 0,
          clips : cl.status==='fulfilled' ? cl.value.files?.length : 0,
          videos: vi.status==='fulfilled' ? vi.value.files?.length : 0,
        })
      } catch {}
    }
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [])

  // Build chart data from confidence history
  const chartData = confHistory.map((p, i) => ({
    i, conf: +(p.v * 100).toFixed(1),
    threshold: 50
  }))

  const pieData = stats?.by_type
    ? Object.entries(stats.by_type).map(([k,v]) => ({ name:k, value:v }))
    : [{ name:'No data', value:1 }]

  const hourlyData = stats?.hourly || []

  return (
    <div style={{ padding:28, display:'flex', flexDirection:'column', gap:24, flex:1 }}>
      {/* Header */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <div>
          <h2 style={{ margin:0, fontSize:22, fontWeight:800 }}>Command Center</h2>
          <p style={{ margin:'4px 0 0', color:'var(--text-muted)', fontSize:14 }}>
            System overview · Real-time analytics
          </p>
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:8, padding:'8px 16px',
                      background:'var(--bg-card)', border:'1px solid var(--border)',
                      borderRadius:10 }}>
          <span className={running ? 'pulse-dot' : ''} style={{
            width:8, height:8, borderRadius:'50%',
            background: running ? (isAnomaly ? 'var(--danger)' : 'var(--success)') : 'var(--text-muted)'
          }}/>
          <span style={{ fontSize:13, fontWeight:600,
            color: running ? (isAnomaly ? 'var(--danger)' : 'var(--success)') : 'var(--text-muted)' }}>
            {running ? (isAnomaly ? 'ANOMALY ACTIVE' : 'MONITORING') : 'SYSTEM IDLE'}
          </span>
        </div>
      </div>

      {/* System cards */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:14 }}>
        <SysCard label="Detection Status" value={running ? 'ACTIVE' : 'IDLE'}
          icon={Activity} color={running ? 'var(--success)' : 'var(--text-muted)'}/>
        <SysCard label="Anomalies Today" value={anomalyCount || stats?.total || 0}
          icon={AlertTriangle} color="var(--danger)"/>
        <SysCard label="Device" value={sysInfo?.device?.toUpperCase() || '…'}
          sub={sysInfo?.gpu || ''} icon={Cpu} color="var(--warning)"/>
        <SysCard label="Output Files" value={counts.videos}
          sub={`${counts.snaps} snaps · ${counts.clips} clips`}
          icon={HardDrive} color="#8b5cf6"/>
      </div>

      {/* Charts row 1 */}
      <div style={{ display:'grid', gridTemplateColumns:'2fr 1fr', gap:20 }}>
        {/* Live confidence chart */}
        <div className="card" style={{ padding:20 }}>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:18 }}>
            <span style={{ fontSize:14, fontWeight:600 }}>Live Confidence Stream</span>
            <span style={{ fontSize:11, color:'var(--text-muted)', fontFamily:'var(--font-mono)' }}>
              Last {chartData.length} frames
            </span>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#0ea5e9" stopOpacity={0.25}/>
                  <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis dataKey="i" hide/>
              <YAxis domain={[0,100]} hide/>
              <Tooltip content={<CustomTooltip/>}/>
              <Line dataKey="threshold" stroke="rgba(239,68,68,0.4)"
                strokeDasharray="4 4" dot={false} strokeWidth={1}/>
              <Area dataKey="conf" stroke="#0ea5e9" fill="url(#confGrad)"
                strokeWidth={2} dot={false} name="Confidence %"/>
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Anomaly type distribution */}
        <div className="card" style={{ padding:20 }}>
          <span style={{ fontSize:14, fontWeight:600, display:'block', marginBottom:18 }}>
            Anomaly Distribution
          </span>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={70}
                dataKey="value" paddingAngle={3}>
                {pieData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]}/>
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip/>}/>
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display:'flex', flexWrap:'wrap', gap:'4px 14px', marginTop:8 }}>
            {pieData.slice(0,4).map((d,i) => (
              <div key={i} style={{ display:'flex', alignItems:'center', gap:5 }}>
                <div style={{ width:8,height:8,borderRadius:'50%',background:COLORS[i%COLORS.length], flexShrink:0}}/>
                <span style={{ fontSize:11, color:'var(--text-muted)' }}>{d.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Charts row 2 */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:20 }}>
        {/* Hourly bar */}
        <div className="card" style={{ padding:20 }}>
          <span style={{ fontSize:14, fontWeight:600, display:'block', marginBottom:18 }}>
            Hourly Anomalies
          </span>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={hourlyData.length ? hourlyData : [{hour:'--',count:0}]}>
              <XAxis dataKey="hour" tick={{ fontSize:11, fill:'var(--text-muted)' }}/>
              <YAxis tick={{ fontSize:11, fill:'var(--text-muted)' }}/>
              <Tooltip content={<CustomTooltip/>}/>
              <Bar dataKey="count" fill="var(--accent)" radius={[4,4,0,0]} name="Events"/>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Output counts */}
        <div className="card" style={{ padding:20 }}>
          <span style={{ fontSize:14, fontWeight:600, display:'block', marginBottom:14 }}>
            Saved Evidence
          </span>
          {[
            { label:'Output Videos', count:counts.videos, icon:Video, color:'var(--accent)' },
            { label:'Snapshots',      count:counts.snaps,  icon:Image, color:'var(--warning)' },
            { label:'Anomaly Clips',  count:counts.clips,  icon:Film,  color:'var(--danger)' },
          ].map(({ label, count, icon:Icon, color }) => (
            <div key={label} style={{ display:'flex', alignItems:'center', gap:12,
                                      padding:'10px 0', borderBottom:'1px solid rgba(56,189,248,0.08)' }}>
              <Icon size={16} color={color}/>
              <span style={{ fontSize:13, flex:1, color:'var(--text-secondary)' }}>{label}</span>
              <span style={{ fontSize:16, fontWeight:700, fontFamily:'var(--font-mono)', color }}>{count}</span>
            </div>
          ))}
        </div>

        {/* Backend status */}
        <div className="card" style={{ padding:20 }}>
          <span style={{ fontSize:14, fontWeight:600, display:'block', marginBottom:14 }}>
            System Health
          </span>
          {[
            { key:'Backend Modules', val: sysInfo?.backend_loaded ? 'Loaded' : 'Partial', ok: sysInfo?.backend_loaded },
            { key:'AI Models',       val: sysInfo?.models_loaded  ? 'Ready'  : 'Pending', ok: sysInfo?.models_loaded },
            { key:'GPU Acceleration',val: sysInfo?.device === 'cuda' ? 'Active' : 'CPU mode', ok: sysInfo?.device==='cuda' },
            { key:'Python Version',  val: sysInfo?.python || '…', ok: true },
          ].map(({ key, val, ok }) => (
            <div key={key} style={{ display:'flex', justifyContent:'space-between',
                                    alignItems:'center', padding:'8px 0',
                                    borderBottom:'1px solid rgba(56,189,248,0.08)' }}>
              <span style={{ fontSize:12, color:'var(--text-muted)' }}>{key}</span>
              <span style={{ fontSize:12, fontWeight:600, fontFamily:'var(--font-mono)',
                             color: ok ? 'var(--success)' : 'var(--danger)' }}>
                {val}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
