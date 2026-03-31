// frontend/src/pages/Login.jsx
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { authAPI } from '../services/api'
import { useAuthStore } from '../store/useStore'
import toast from 'react-hot-toast'
import { ShieldCheck, Eye, EyeOff, Scan } from 'lucide-react'

export default function Login() {
  const [form, setForm] = useState({ username: 'admin', password: 'admin123' })
  const [show, setShow]   = useState(false)
  const [busy, setBusy]   = useState(false)
  const login             = useAuthStore(s => s.login)
  const nav               = useNavigate()

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    try {
      const data = await authAPI.login(form.username, form.password)
      login({ username: data.username, role: data.role }, data.token)
      toast.success('Access granted')
      nav('/dashboard')
    } catch {
      toast.error('Invalid credentials')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{
      minHeight:'100vh', display:'flex', alignItems:'center',
      justifyContent:'center', background:'var(--bg-base)',
      position:'relative', overflow:'hidden'
    }}>
      {/* Background grid */}
      <div style={{
        position:'absolute', inset:0,
        backgroundImage:`linear-gradient(rgba(14,165,233,0.05) 1px, transparent 1px),
                         linear-gradient(90deg, rgba(14,165,233,0.05) 1px, transparent 1px)`,
        backgroundSize:'40px 40px',
        pointerEvents:'none'
      }}/>
      {/* Glow */}
      <div style={{
        position:'absolute', top:'20%', left:'50%', transform:'translateX(-50%)',
        width:600, height:300,
        background:'radial-gradient(ellipse, rgba(14,165,233,0.12) 0%, transparent 70%)',
        pointerEvents:'none'
      }}/>

      <div className="fade-in" style={{ width:'100%', maxWidth:400, padding:'0 20px' }}>
        {/* Logo */}
        <div style={{ textAlign:'center', marginBottom:40 }}>
          <div style={{
            width:64, height:64, borderRadius:16,
            background:'linear-gradient(135deg, #0ea5e9 0%, #0369a1 100%)',
            display:'flex', alignItems:'center', justifyContent:'center',
            margin:'0 auto 16px', boxShadow:'0 0 32px rgba(14,165,233,0.4)'
          }}>
            <Scan size={32} color="#fff"/>
          </div>
          <h1 style={{ margin:0, fontSize:24, fontWeight:800, letterSpacing:-0.5 }}>
            SurveillanceAI
          </h1>
          <p style={{ margin:'6px 0 0', color:'var(--text-muted)', fontSize:13 }}>
            Anomaly Detection System v2.0
          </p>
        </div>

        {/* Card */}
        <div className="card" style={{ padding:32 }}>
          <p style={{ margin:'0 0 24px', fontSize:12, color:'var(--text-muted)',
                      letterSpacing:.08em, textTransform:'uppercase', fontFamily:'var(--font-mono)' }}>
            System Access
          </p>
          <form onSubmit={submit}>
            <div style={{ marginBottom:16 }}>
              <label style={{ display:'block', fontSize:12, color:'var(--text-muted)',
                              marginBottom:6, letterSpacing:.05em }}>USERNAME</label>
              <input className="input-base"
                value={form.username}
                onChange={e=>setForm({...form,username:e.target.value})}
                placeholder="admin"
                autoComplete="username"/>
            </div>
            <div style={{ marginBottom:24, position:'relative' }}>
              <label style={{ display:'block', fontSize:12, color:'var(--text-muted)',
                              marginBottom:6, letterSpacing:.05em }}>PASSWORD</label>
              <input className="input-base" style={{ paddingRight:40 }}
                type={show ? 'text' : 'password'}
                value={form.password}
                onChange={e=>setForm({...form,password:e.target.value})}
                placeholder="••••••••"
                autoComplete="current-password"/>
              <button type="button" onClick={()=>setShow(!show)} style={{
                position:'absolute', right:12, bottom:10, background:'none',
                border:'none', color:'var(--text-muted)', cursor:'pointer', padding:0
              }}>
                {show ? <EyeOff size={16}/> : <Eye size={16}/>}
              </button>
            </div>
            <button className="btn-primary" type="submit"
              disabled={busy} style={{ width:'100%', padding:'12px 20px', fontSize:14 }}>
              {busy ? 'Authenticating…' : 'Access System'}
            </button>
          </form>
          <p style={{ textAlign:'center', marginTop:20, fontSize:12, color:'var(--text-muted)' }}>
            Demo: <span style={{fontFamily:'var(--font-mono)'}}>admin / admin123</span>
          </p>
        </div>

        {/* Footer */}
        <div style={{ textAlign:'center', marginTop:24, display:'flex',
                      alignItems:'center', justifyContent:'center', gap:6 }}>
          <ShieldCheck size={14} color="var(--success)"/>
          <span style={{ fontSize:12, color:'var(--text-muted)' }}>
            Encrypted · AI-Powered · Real-time
          </span>
        </div>
      </div>
    </div>
  )
}
