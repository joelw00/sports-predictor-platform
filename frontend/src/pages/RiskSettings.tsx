import { useEffect, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api, RiskDecision, RiskPolicy } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'

const NUM_FIELDS: Array<{
  key: keyof Omit<RiskPolicy, 'id' | 'name' | 'enabled'>
  label: string
  help: string
  step: number
  min: number
  max?: number
}> = [
  { key: 'bankroll', label: 'Bankroll', help: 'Total stake budget in units', step: 10, min: 0 },
  { key: 'kelly_fraction', label: 'Fractional Kelly', help: '0.25 = quarter-Kelly (conservative)', step: 0.05, min: 0, max: 1 },
  { key: 'max_stake_pct', label: 'Max stake per bet', help: 'Hard cap as % of bankroll', step: 0.005, min: 0, max: 1 },
  { key: 'max_daily_exposure_pct', label: 'Max daily exposure', help: 'Hard cap across all bets in one day', step: 0.01, min: 0, max: 1 },
  { key: 'max_concurrent_positions', label: 'Max concurrent positions', help: 'Cap across open exposures', step: 1, min: 0 },
  { key: 'stop_loss_drawdown_pct', label: 'Stop loss drawdown', help: 'Halt betting when realised drawdown exceeds', step: 0.01, min: 0, max: 1 },
  { key: 'min_edge', label: 'Min edge', help: 'Skip bets below this edge', step: 0.005, min: 0 },
  { key: 'min_confidence', label: 'Min confidence', help: 'Skip bets below this confidence', step: 0.05, min: 0, max: 1 },
]

export default function RiskSettingsPage() {
  const qc = useQueryClient()
  const { data: policy } = useQuery({ queryKey: ['risk', 'policy'], queryFn: api.riskPolicy })
  const [draft, setDraft] = useState<RiskPolicy | null>(null)
  const [saving, setSaving] = useState(false)
  const [savedAt, setSavedAt] = useState<number | null>(null)

  useEffect(() => {
    if (policy && !draft) setDraft(policy)
  }, [policy, draft])

  const decisions = useQuery({ queryKey: ['risk', 'evaluate'], queryFn: () => api.riskEvaluate() })

  const dirty =
    draft && policy && NUM_FIELDS.some((f) => draft[f.key] !== policy[f.key]) || draft?.enabled !== policy?.enabled

  async function save() {
    if (!draft) return
    setSaving(true)
    try {
      const { id: _id, name: _name, ...body } = draft
      void _id
      void _name
      const updated = await api.updateRiskPolicy(body)
      qc.setQueryData(['risk', 'policy'], updated)
      setDraft(updated)
      setSavedAt(Date.now())
      await qc.invalidateQueries({ queryKey: ['risk', 'evaluate'] })
    } finally {
      setSaving(false)
    }
  }

  if (!draft) return <div className="text-muted-foreground text-sm">Loading policy…</div>

  const accepted = decisions.data?.filter((d) => d.accepted) ?? []
  const rejected = decisions.data?.filter((d) => !d.accepted) ?? []
  const totalStake = accepted.reduce((s, d) => s + d.recommended_stake, 0)

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Risk policy</CardTitle>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={draft.enabled}
                onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })}
              />
              enabled
            </label>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {NUM_FIELDS.map((f) => (
              <label key={f.key} className="block text-sm">
                <div className="font-medium">{f.label}</div>
                <div className="text-xs text-muted-foreground mb-1">{f.help}</div>
                <Input
                  type="number"
                  step={f.step}
                  min={f.min}
                  max={f.max}
                  value={draft[f.key] as number}
                  onChange={(e) =>
                    setDraft({ ...draft, [f.key]: Number(e.target.value) })
                  }
                />
              </label>
            ))}
          </div>
          <div className="mt-4 flex items-center gap-3">
            <Button disabled={!dirty || saving} onClick={save}>
              {saving ? 'Saving…' : 'Save policy'}
            </Button>
            {savedAt && (
              <span className="text-xs text-muted-foreground">
                saved · changes applied to new evaluations
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Current value bets, filtered by policy</CardTitle>
            <div className="text-xs text-muted-foreground">
              {accepted.length} accepted · {rejected.length} rejected · total stake{' '}
              <span className="tabular-nums">{totalStake.toFixed(2)}</span>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {decisions.isLoading && (
            <div className="p-4 text-sm italic text-muted-foreground">computing…</div>
          )}
          {decisions.data && decisions.data.length === 0 && (
            <div className="p-4 text-sm italic text-muted-foreground">no value bets right now</div>
          )}
          {decisions.data && decisions.data.length > 0 && <DecisionTable rows={decisions.data} />}
        </CardContent>
      </Card>
    </div>
  )
}

function DecisionTable({ rows }: { rows: RiskDecision[] }) {
  return (
    <table className="w-full text-sm">
      <thead className="text-xs uppercase text-muted-foreground">
        <tr>
          <th className="text-left px-4 py-2">Market</th>
          <th className="text-left px-4 py-2">Book</th>
          <th className="text-right px-4 py-2">Edge</th>
          <th className="text-right px-4 py-2">Kelly</th>
          <th className="text-right px-4 py-2">Stake</th>
          <th className="text-left px-4 py-2">Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((d, i) => (
          <tr key={i} className="border-t border-border">
            <td className="px-4 py-2">
              <span className="text-muted-foreground">#{d.match_id}</span> {d.market} · {d.selection}
              {d.line != null ? ` @ ${d.line}` : ''}
            </td>
            <td className="px-4 py-2">{d.bookmaker}</td>
            <td
              className={`px-4 py-2 text-right font-semibold tabular-nums ${
                d.edge > 0 ? 'text-[hsl(var(--success))]' : 'text-destructive'
              }`}
            >
              {(d.edge * 100).toFixed(1)}%
            </td>
            <td className="px-4 py-2 text-right tabular-nums text-muted-foreground">
              {(d.kelly_fraction * 100).toFixed(1)}%
            </td>
            <td className="px-4 py-2 text-right tabular-nums">
              {d.recommended_stake.toFixed(2)}
            </td>
            <td className="px-4 py-2">
              {d.accepted ? (
                <Badge variant="success">accepted</Badge>
              ) : (
                <span className="flex flex-wrap gap-1">
                  {d.reasons.map((r) => (
                    <Badge key={r} variant="warning" className="text-[10px]">
                      {r}
                    </Badge>
                  ))}
                </span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
