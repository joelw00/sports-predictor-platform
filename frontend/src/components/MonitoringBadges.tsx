import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, Info, ShieldAlert } from 'lucide-react'
import { api, type MonitoringAlert, type MonitoringSnapshot } from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'

const SEVERITY_ORDER: Record<MonitoringAlert['severity'], number> = {
  critical: 0,
  warning: 1,
  info: 2,
}

function variantForSeverity(sev: MonitoringAlert['severity']) {
  if (sev === 'critical') return 'destructive' as const
  if (sev === 'warning') return 'warning' as const
  return 'secondary' as const
}

function iconForSeverity(sev: MonitoringAlert['severity']) {
  if (sev === 'critical') return ShieldAlert
  if (sev === 'warning') return AlertTriangle
  return Info
}

function labelForCode(code: string): string {
  switch (code) {
    case 'low_data':
      return 'Low data'
    case 'high_drift':
      return 'Feature drift'
    case 'stale_model':
      return 'Stale model'
    case 'calibration_drift':
      return 'Calibration drift'
    default:
      return code
  }
}

function formatComputedAt(iso: string | null): string {
  if (!iso) return 'never'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

export default function MonitoringBadges() {
  const { data, error, isLoading } = useQuery<MonitoringSnapshot | null>({
    queryKey: ['monitoring-latest', 'football', '1x2'],
    queryFn: async () => {
      try {
        return await api.monitoringLatest('football', '1x2')
      } catch (err) {
        // 404 before any monitoring pass ran → null, handled below as a soft state.
        if (err instanceof Error && err.message.startsWith('404')) return null
        throw err
      }
    },
    refetchInterval: 120_000,
  })

  if (isLoading) return null
  if (error) return null

  if (!data) {
    return (
      <Card>
        <CardContent className="flex items-center justify-between gap-3 p-4">
          <div className="flex items-center gap-2">
            <Info className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-sm font-medium">Model health</div>
              <div className="text-xs text-muted-foreground">
                No monitoring pass recorded yet. Trigger one via{' '}
                <code className="rounded bg-muted px-1">POST /monitoring/run</code>.
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  const alerts = [...(data.alerts ?? [])].sort(
    (a, b) => (SEVERITY_ORDER[a.severity] ?? 99) - (SEVERITY_ORDER[b.severity] ?? 99),
  )

  return (
    <Card>
      <CardContent className="flex flex-col gap-3 p-4 md:flex-row md:items-center md:justify-between">
        <div>
          <div className="text-sm font-medium">Model health</div>
          <div className="text-xs text-muted-foreground">
            Live Brier{' '}
            <span className="font-mono tabular-nums">
              {data.brier_live != null ? data.brier_live.toFixed(3) : '—'}
            </span>{' '}
            · training Brier{' '}
            <span className="font-mono tabular-nums">
              {data.brier_training != null ? data.brier_training.toFixed(3) : '—'}
            </span>{' '}
            · max PSI{' '}
            <span className="font-mono tabular-nums">
              {data.max_psi != null ? data.max_psi.toFixed(3) : '—'}
            </span>{' '}
            · last check {formatComputedAt(data.computed_at)}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {alerts.length === 0 ? (
            <Badge variant="success" title="No active alerts from the monitoring pass.">
              Healthy
            </Badge>
          ) : (
            alerts.map((alert) => {
              const Icon = iconForSeverity(alert.severity)
              return (
                <Badge
                  key={`${alert.code}-${alert.severity}`}
                  variant={variantForSeverity(alert.severity)}
                  title={alert.message}
                  className="gap-1"
                >
                  <Icon className="h-3 w-3" />
                  {labelForCode(alert.code)}
                </Badge>
              )
            })
          )}
        </div>
      </CardContent>
    </Card>
  )
}
