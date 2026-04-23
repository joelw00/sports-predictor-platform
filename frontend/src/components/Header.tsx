import { NavLink } from 'react-router-dom'
import { Activity, BarChart3, Gauge, Home, Info, Trophy } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'

const NAV = [
  { to: '/', label: 'Dashboard', icon: Home },
  { to: '/value-bets', label: 'Value Bets', icon: Trophy },
  { to: '/backtest', label: 'Backtest', icon: BarChart3 },
  { to: '/monitoring', label: 'Monitoring', icon: Gauge },
  { to: '/about', label: 'About', icon: Info },
]

export default function Header() {
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 60_000 })
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur">
      <div className="container flex h-14 items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="h-5 w-5 text-primary" />
          <span className="font-semibold tracking-tight">Sports Predictor</span>
          {health != null &&
            (health.demo_mode ? (
              <Badge
                variant="warning"
                title="No external data source configured — using synthetic demo data."
              >
                Demo mode
              </Badge>
            ) : (
              <Badge
                variant="success"
                title="At least one real data adapter is active (e.g. Football-Data.org)."
              >
                Real data
              </Badge>
            ))}
        </div>
        <nav className="flex items-center gap-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-secondary text-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary/60',
                )
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
