import { Route, Routes } from 'react-router-dom'
import Header from '@/components/Header'
import Dashboard from '@/pages/Dashboard'
import ValueBetsPage from '@/pages/ValueBets'
import MatchDetailPage from '@/pages/MatchDetail'
import BacktestPage from '@/pages/Backtest'
import MonitoringPage from '@/pages/Monitoring'
import About from '@/pages/About'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1 container py-6 animate-fade-in">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/value-bets" element={<ValueBetsPage />} />
          <Route path="/matches/:id" element={<MatchDetailPage />} />
          <Route path="/backtest" element={<BacktestPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
      <footer className="border-t border-border py-4 text-center text-xs text-muted-foreground">
        Sports Predictor Platform · probabilistic analytics — never a guarantee.
      </footer>
    </div>
  )
}
