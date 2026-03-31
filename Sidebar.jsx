// frontend/src/components/dashboard/Sidebar.jsx
import { NavLink } from 'react-router-dom'
import { useAuthStore } from '../../store/useStore'
import { useNavigate } from 'react-router-dom'
import {
  LayoutDashboard, Video, Wifi, Bell, FileText,
  Image, Film, Settings, LogOut, Scan, Activity
} from 'lucide-react'

const NAV = [
  { to:'/dashboard',           icon: LayoutDashboard, label:'Dashboard' },
  { to:'/dashboard/detection', icon: Video,           label:'Detection' },
  { to:'/dashboard/live',      icon: Wifi,            label:'Live / RTSP' },
  { to:'/dashboard/alerts',    icon: Bell,            label:'Alerts' },
  { to:'/dashboard/logs',      icon: FileText,        label:'Logs' },
  { to:'/dashboard/snapshots', icon: Image,           label:'Snapshots' },
  { to:'/dashboard/clips',     icon: Film,            label:'Anomaly Clips' },
  { to:'/dashboard/analytics', icon: Activity,        label:'Analytics' },
]

const linkStyle = (active) => ({
  display:'flex', alignItems:'center', gap:12, padding:'10px 16px',
  borderRadius:10, textDecoration:'none', fontSize:14, fontWeight:500,
  color: active ? '#fff' : 'var(--text-muted)',
  background: active ? 'rgba(14,165,233,0.18)' : 'transparent',
  borderLeft: active ? '2px solid var(--accent)' : '2px solid transparent',
  transition:'all 0.15s', marginBottom:2,
})

export default function Sidebar() {
  const logout = useAuthStore(s => s.logout)
  const user   = useAuthStore(s => s.user)
  const nav    = useNavigate()

  const doLogout = () => { logout(); nav('/login') }

  return (
    <aside style={{
      width:230, minHeight:'100vh', background:'var(--bg-surface)',
      borderRight:'1px solid var(--border)', display:'flex',
      flexDirection:'column', padding:'20px 12px', flexShrink:0
    }}>
      {/* Logo */}
      <div style={{ display:'flex', alignItems:'center', gap:10, padding:'4px 8px', marginBottom:28 }}>
        <div style={{
          width:34, height:34, borderRadius:8, flexShrink:0,
          background:'linear-gradient(135deg,#0ea5e9,#0369a1)',
          display:'flex', alignItems:'center', justifyContent:'center',
          boxShadow:'0 0 16px rgba(14,165,233,0.35)'
        }}>
          <Scan size={18} color="#fff"/>
        </div>
        <div>
          <div style={{ fontSize:14, fontWeight:800, lineHeight:1.2 }}>Surveillance</div>
          <div style={{ fontSize:10, color:'var(--text-muted)', fontFamily:'var(--font-mono)', letterSpacing:.05em }}>AI v2.0</div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex:1 }}>
        <p style={{ fontSize:10, color:'var(--text-muted)', letterSpacing:.12em,
                    textTransform:'uppercase', padding:'0 8px', marginBottom:8 }}>
          Navigation
        </p>
        {NAV.map(({ to, icon:Icon, label }) => (
          <NavLink key={to} to={to} end={to==='/dashboard'}>
            {({ isActive }) => (
              <div style={linkStyle(isActive)}>
                <Icon size={16} style={{ flexShrink:0 }}/>
                <span>{label}</span>
              </div>
            )}
          </NavLink>
        ))}
      </nav>

      {/* User + logout */}
      <div style={{ borderTop:'1px solid var(--border)', paddingTop:16 }}>
        <div style={{ display:'flex', alignItems:'center', gap:10, padding:'0 8px', marginBottom:12 }}>
          <div style={{
            width:30, height:30, borderRadius:50, flexShrink:0,
            background:'var(--accent-dim)', display:'flex',
            alignItems:'center', justifyContent:'center',
            fontSize:12, fontWeight:700
          }}>
            {user?.username?.[0]?.toUpperCase() || 'A'}
          </div>
          <div>
            <div style={{ fontSize:13, fontWeight:600 }}>{user?.username || 'Admin'}</div>
            <div style={{ fontSize:11, color:'var(--text-muted)' }}>{user?.role || 'admin'}</div>
          </div>
        </div>
        <button onClick={doLogout} style={{
          ...linkStyle(false), width:'100%', border:'none', cursor:'pointer',
          background:'none', textAlign:'left'
        }}>
          <LogOut size={16}/> <span>Logout</span>
        </button>
      </div>
    </aside>
  )
}
