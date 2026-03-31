// frontend/src/App.jsx
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { useAuthStore } from './store/useStore'
import Sidebar from './components/dashboard/Sidebar'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Detection from './pages/Detection'
import Analytics from './pages/Analytics'
import Live from './pages/Live'
import Logs from './pages/Logs'
import { Snapshots, Clips, Alerts } from './pages/OtherPages'
import './styles/globals.css'

function RequireAuth() {
  const token = useAuthStore(s => s.token)
  return token ? <Outlet/> : <Navigate to="/login" replace/>
}

function DashboardLayout() {
  return (
    <div style={{ display:'flex', minHeight:'100vh' }}>
      <Sidebar/>
      <main style={{ flex:1, overflowY:'auto', background:'var(--bg-base)' }}>
        <Outlet/>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: '#111827',
            color: '#f0f6ff',
            border: '1px solid rgba(56,189,248,0.2)',
            fontFamily: "'Syne', sans-serif",
            fontSize: 13
          }
        }}
      />
      <Routes>
        <Route path="/login" element={<Login/>}/>
        <Route path="/" element={<Navigate to="/dashboard" replace/>}/>
        <Route element={<RequireAuth/>}>
          <Route element={<DashboardLayout/>}>
            <Route path="/dashboard"            element={<Dashboard/>}/>
            <Route path="/dashboard/detection"  element={<Detection/>}/>
            <Route path="/dashboard/live"       element={<Live/>}/>
            <Route path="/dashboard/alerts"     element={<Alerts/>}/>
            <Route path="/dashboard/logs"       element={<Logs/>}/>
            <Route path="/dashboard/snapshots"  element={<Snapshots/>}/>
            <Route path="/dashboard/clips"      element={<Clips/>}/>
            <Route path="/dashboard/analytics"  element={<Analytics/>}/>
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
