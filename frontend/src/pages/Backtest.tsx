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
import { api, BacktestResult } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { formatPercent } from '@/lib/utils'

export default function BacktestPage() {
  const [sport, setSport] = useState('football')
  const [market, setMarket] = useState('1x2')
  const [minEdge, setMinEdge] = useState('0.03')
  const [strategy, setStrategy] = useState('flat')
  const [stake, setStake] = useState('1')
  const [mode, setMode] = useState<'walk_forward' | 'pretrained'>('walk_forward')
  const qc = useQueryClient()

  const { data: runs } = useQuery({ queryKey: ['backtests'], queryFn: api.backtests })

  const mutation = useMutation({
    mutationFn: () =>
      api.runBacktest({
        sport,
        market,
        strategy,
        mode,
        stake: Number(stake),
        min_edge: Number(minEdge),
        label: `${mode}/${sport}/${market}/${strategy}`,
      } as unknown as Partial<BacktestResult>),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['backtests'] }),
  })

  const last = useMemo(() => runs?.[0], [runs])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Backtesting</h1>
        <p className="text-sm text-muted-foreground">
          Replay history using closing odds and the current predictor to measure ROI, yield and drawdown.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>New simulation</CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            <span className="font-medium">Walk-forward</span> retrains the predictor at every
            fold boundary and scores only future matches — this is the honest evaluation.
            <span className="font-medium"> Pretrained</span> replays using the already-trained
            predictor (leaky; kept only for before/after comparison).
          </p>
        </CardHeader>
        <CardContent className="grid grid-cols-2 md:grid-cols-7 gap-3 items-end">
          <Field label="Mode">
            <Select value={mode} onValueChange={(v) => setMode(v as 'walk_forward' | 'pretrained')}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="walk_forward">Walk-forward</SelectItem>
                <SelectItem value="pretrained">Pretrained (leaky)</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <Field label="Sport">
            <Select value={sport} onValueChange={setSport}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="football">Football</SelectItem>
                <SelectItem value="table_tennis">Table Tennis</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <Field label="Market">
            <Select value={market} onValueChange={setMarket}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="1x2">1X2</SelectItem>
                <SelectItem value="over_under">Over / Under</SelectItem>
                <SelectItem value="btts">BTTS</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <Field label="Strategy">
            <Select value={strategy} onValueChange={setStrategy}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="flat">Flat stake</SelectItem>
                <SelectItem value="kelly">Fractional Kelly</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <Field label="Stake"><Input type="number" step="0.1" value={stake} onChange={(e) => setStake(e.target.value)} /></Field>
          <Field label="Min edge"><Input type="number" step="0.005" value={minEdge} onChange={(e) => setMinEdge(e.target.value)} /></Field>
          <Button onClick={() => mutation.mutate()} disabled={mutation.isPending}>
            {mutation.isPending ? 'Running…' : 'Run backtest'}
          </Button>
        </CardContent>
      </Card>

      {last && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>{last.label}</CardTitle>
              <div className="text-xs text-muted-foreground">
                {last.sport_code} · {last.market} · {last.strategy} · stake {last.stake}
              </div>
            </div>
          </CardHeader>
          <CardContent className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <Stat label="Bets" value={last.n_bets} />
            <Stat label="Hit rate" value={formatPercent(last.n_bets ? last.n_wins / last.n_bets : 0)} />
            <Stat label="ROI" value={formatPercent(last.roi)} tone={last.roi >= 0 ? 'success' : 'danger'} />
            <Stat label="Yield" value={`${last.yield_pct.toFixed(2)}%`} />
            <Stat label="Max drawdown" value={last.max_drawdown.toFixed(2)} />
          </CardContent>
          {last.breakdown && typeof last.breakdown === 'object' ? (
            <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-3 border-t border-border">
              <Stat
                label="Mode"
                value={String((last.breakdown as { mode?: string }).mode ?? 'pretrained')}
              />
              <Stat
                label="Brier (1X2)"
                value={(() => {
                  const c = (last.breakdown as { calibration?: { brier?: number } }).calibration
                  return c?.brier !== undefined ? c.brier.toFixed(4) : '—'
                })()}
              />
              <Stat
                label="Log-loss (holdout)"
                value={(() => {
                  const c = (last.breakdown as { calibration?: { log_loss?: number } }).calibration
                  return c?.log_loss !== undefined ? c.log_loss.toFixed(4) : '—'
                })()}
              />
              <Stat
                label="Holdout matches"
                value={(() => {
                  const c = (last.breakdown as { calibration?: { n_holdout?: number } }).calibration
                  return c?.n_holdout ?? '—'
                })()}
              />
            </CardContent>
          ) : null}
          {last.equity_curve?.length ? (
            <CardContent className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={last.equity_curve}>
                  <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={(v) => v.slice(5, 10)} stroke="hsl(var(--muted-foreground))" fontSize={11} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
                  <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8 }} />
                  <Line type="monotone" dataKey="bankroll" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          ) : null}
        </Card>
      )}

      <Card>
        <CardHeader><CardTitle>Past runs</CardTitle></CardHeader>
        <CardContent className="p-0">
          <table className="w-full text-sm">
            <thead className="text-xs uppercase text-muted-foreground">
              <tr>
                <th className="text-left px-4 py-2">Label</th>
                <th className="text-left px-4 py-2">Sport</th>
                <th className="text-left px-4 py-2">Market</th>
                <th className="text-right px-4 py-2">Bets</th>
                <th className="text-right px-4 py-2">ROI</th>
                <th className="text-right px-4 py-2">Max DD</th>
              </tr>
            </thead>
            <tbody>
              {(runs ?? []).map((r) => (
                <tr key={r.id} className="border-t border-border">
                  <td className="px-4 py-2">{r.label}</td>
                  <td className="px-4 py-2">{r.sport_code}</td>
                  <td className="px-4 py-2">{r.market}</td>
                  <td className="px-4 py-2 text-right tabular-nums">{r.n_bets}</td>
                  <td className={`px-4 py-2 text-right tabular-nums ${r.roi >= 0 ? 'text-[hsl(var(--success))]' : 'text-destructive'}`}>
                    {formatPercent(r.roi)}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums">{r.max_drawdown.toFixed(2)}</td>
                </tr>
              ))}
              {(!runs || runs.length === 0) && (
                <tr>
                  <td colSpan={6} className="px-4 py-6 text-center text-muted-foreground italic">
                    no backtests yet — run one above
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

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <label className="text-xs text-muted-foreground">{label}</label>
      {children}
    </div>
  )
}

function Stat({ label, value, tone }: { label: string; value: string | number; tone?: 'success' | 'danger' }) {
  const toneClass = tone === 'success' ? 'text-[hsl(var(--success))]' : tone === 'danger' ? 'text-destructive' : ''
  return (
    <div className="rounded-md border border-border bg-card p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`text-xl font-semibold tabular-nums ${toneClass}`}>{value}</div>
    </div>
  )
}
