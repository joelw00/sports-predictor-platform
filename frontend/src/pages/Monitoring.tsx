import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AlertTriangle, Info, RefreshCw, ShieldAlert } from 'lucide-react'
import { api, type MonitoringAlert, type MonitoringSnapshot } from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

const MARKETS = ['1x2', 'btts', 'over_2_5'] as const
type Market = (typeof MARKETS)[number]

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

function psiTone(psi: number | null | undefined): 'danger' | 'warning' | 'muted' {
  if (psi == null) return 'muted'
  if (psi >= 0.25) return 'danger'
  if (psi >= 0.1) return 'warning'
  return 'muted'
}

export default function MonitoringPage() {
  const [market, setMarket] = useState<Market>('1x2')
  const qc = useQueryClient()

  const latestQuery = useQuery<MonitoringSnapshot | null>({
    queryKey: ['monitoring-latest', 'football', market],
    queryFn: async () => {
      try {
        return await api.monitoringLatest('football', market)
      } catch (err) {
        if (err instanceof Error && err.message.startsWith('404')) return null
        throw err
      }
    },
    refetchInterval: 120_000,
  })

  const historyQuery = useQuery<MonitoringSnapshot[]>({
    queryKey: ['monitoring-history', 'football', market],
    queryFn: () => api.monitoringHistory('football', market, 60),
    refetchInterval: 120_000,
  })

  const runMutation = useMutation({
    mutationFn: () => api.runMonitoring('football', market),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['monitoring-latest', 'football', market] })
      qc.invalidateQueries({ queryKey: ['monitoring-history', 'football', market] })
    },
  })

  const latest = latestQuery.data

  const driftRows = useMemo(() => {
    const features = latest?.drift?.features ?? []
    // Sort by PSI desc so the worst offenders land on top.
    return [...features].sort((a, b) => (b.psi ?? 0) - (a.psi ?? 0))
  }, [latest])

  const history = useMemo(() => {
    const rows = historyQuery.data ?? []
    // API returns newest first; charts expect chronological order.
    return [...rows].reverse().map((s) => ({
      date: s.computed_at ?? '',
      brier_live: s.brier_live,
      brier_training: s.brier_training,
      max_psi: s.max_psi,
    }))
  }, [historyQuery.data])

  const alerts = useMemo(
    () =>
      [...(latest?.alerts ?? [])].sort(
        (a, b) => (SEVERITY_ORDER[a.severity] ?? 99) - (SEVERITY_ORDER[b.severity] ?? 99),
      ),
    [latest],
  )

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Monitoring</h1>
          <p className="text-sm text-muted-foreground">
            Live Brier tracker and feature-level drift (PSI / KS) for the active production model.
            Snapshots are written daily by the scheduler and on demand via the button below.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={market} onValueChange={(v) => setMarket(v as Market)}>
            <SelectTrigger className="w-40"><SelectValue /></SelectTrigger>
            <SelectContent>
              {MARKETS.map((mk) => (
                <SelectItem key={mk} value={mk}>{mk.toUpperCase()}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="secondary"
            onClick={() => runMutation.mutate()}
            disabled={runMutation.isPending}
            className="gap-2"
          >
            <RefreshCw className={runMutation.isPending ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
            {runMutation.isPending ? 'Running…' : 'Run snapshot'}
          </Button>
        </div>
      </div>

      {latest ? (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Latest snapshot · {market.toUpperCase()}</CardTitle>
              <div className="text-xs text-muted-foreground">
                {latest.computed_at ? new Date(latest.computed_at).toLocaleString() : '—'}
                {latest.model_version ? ` · model ${latest.model_version}` : ''}
              </div>
            </div>
          </CardHeader>
          <CardContent className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <Stat
              label="Brier live"
              value={latest.brier_live != null ? latest.brier_live.toFixed(4) : '—'}
            />
            <Stat
              label="Brier training"
              value={latest.brier_training != null ? latest.brier_training.toFixed(4) : '—'}
            />
            <Stat
              label="Log-loss live"
              value={latest.log_loss_live != null ? latest.log_loss_live.toFixed(4) : '—'}
            />
            <Stat
              label="Max PSI"
              value={latest.max_psi != null ? latest.max_psi.toFixed(3) : '—'}
              tone={psiTone(latest.max_psi)}
            />
            <Stat label="Evaluated matches" value={String(latest.n_predictions_evaluated)} />
          </CardContent>
          <CardContent className="flex flex-wrap gap-2 border-t border-border">
            {alerts.length === 0 ? (
              <Badge variant="success" title="No active alerts.">Healthy</Badge>
            ) : (
              alerts.map((a) => {
                const Icon = iconForSeverity(a.severity)
                return (
                  <Badge
                    key={`${a.code}-${a.severity}`}
                    variant={variantForSeverity(a.severity)}
                    title={a.message}
                    className="gap-1"
                  >
                    <Icon className="h-3 w-3" />
                    {labelForCode(a.code)}
                  </Badge>
                )
              })
            )}
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="p-4 text-sm text-muted-foreground">
            No monitoring snapshot recorded for <code className="rounded bg-muted px-1">{market}</code>{' '}
            yet. Click <span className="font-medium">Run snapshot</span> to compute one.
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader><CardTitle>Brier history</CardTitle></CardHeader>
        <CardContent className="h-72">
          {history.length === 0 ? (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground italic">
              no history yet
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(v: string) => (v ? v.slice(5, 10) : '')}
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={11}
                />
                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    background: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: 8,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="brier_live"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={false}
                  name="Brier (live)"
                />
                <Line
                  type="monotone"
                  dataKey="brier_training"
                  stroke="hsl(var(--muted-foreground))"
                  strokeDasharray="4 4"
                  strokeWidth={2}
                  dot={false}
                  name="Brier (training)"
                />
                <Line
                  type="monotone"
                  dataKey="max_psi"
                  stroke="hsl(var(--destructive))"
                  strokeWidth={2}
                  dot={false}
                  name="Max PSI"
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Feature drift (PSI / KS)</CardTitle></CardHeader>
        <CardContent className="p-0">
          <table className="w-full text-sm">
            <thead className="text-xs uppercase text-muted-foreground">
              <tr>
                <th className="text-left px-4 py-2">Feature</th>
                <th className="text-right px-4 py-2">PSI</th>
                <th className="text-right px-4 py-2">KS</th>
                <th className="text-right px-4 py-2">KS p-value</th>
                <th className="text-right px-4 py-2">n ref / cur</th>
              </tr>
            </thead>
            <tbody>
              {driftRows.map((row) => {
                const tone = psiTone(row.psi)
                const toneClass =
                  tone === 'danger'
                    ? 'text-destructive'
                    : tone === 'warning'
                      ? 'text-[hsl(var(--warning-foreground,0_0%_10%))]'
                      : ''
                return (
                  <tr key={row.feature} className="border-t border-border">
                    <td className="px-4 py-2 font-mono text-xs">{row.feature}</td>
                    <td className={`px-4 py-2 text-right tabular-nums ${toneClass}`}>
                      {row.psi != null ? row.psi.toFixed(3) : '—'}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {row.ks_statistic != null ? row.ks_statistic.toFixed(3) : '—'}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {row.ks_pvalue != null ? row.ks_pvalue.toFixed(3) : '—'}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums text-muted-foreground">
                      {row.n_ref} / {row.n_cur}
                    </td>
                  </tr>
                )
              })}
              {driftRows.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-6 text-center text-muted-foreground italic">
                    no feature drift recorded — run a snapshot to populate this table
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  )
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string
  value: string
  tone?: 'danger' | 'warning' | 'muted'
}) {
  const toneClass =
    tone === 'danger'
      ? 'text-destructive'
      : tone === 'warning'
        ? 'text-[hsl(var(--warning-foreground,0_0%_10%))]'
        : ''
  return (
    <div className="rounded-md border border-border bg-card p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`text-xl font-semibold tabular-nums ${toneClass}`}>{value}</div>
    </div>
  )
}
