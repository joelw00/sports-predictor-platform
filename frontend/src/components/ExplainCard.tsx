import { useQuery } from '@tanstack/react-query'
import { api, OutcomeExplanation } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { formatPercent } from '@/lib/utils'

interface Props {
  matchId: number
}

const OUTCOME_LABEL: Record<string, string> = {
  home: 'Home win',
  draw: 'Draw',
  away: 'Away win',
}

export default function ExplainCard({ matchId }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['explain', matchId],
    queryFn: () => api.explainMatch(matchId, 5),
    enabled: Number.isFinite(matchId),
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>Why this prediction (SHAP)</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        {isLoading && <div className="px-4 py-4 text-sm italic text-muted-foreground">computing…</div>}
        {error && (
          <div className="px-4 py-4 text-sm italic text-muted-foreground">
            explanation unavailable (untrained model?)
          </div>
        )}
        {data && (
          <div className="divide-y divide-border">
            {data.outcomes.map((o) => (
              <OutcomeBlock key={o.outcome} outcome={o} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function OutcomeBlock({ outcome }: { outcome: OutcomeExplanation }) {
  const all = [
    ...outcome.top_positive.map((c) => ({ ...c, sign: 'pos' as const })),
    ...outcome.top_negative.map((c) => ({ ...c, sign: 'neg' as const })),
  ]
  const max = Math.max(1e-6, ...all.map((c) => Math.abs(c.shap_value)))

  return (
    <details className="group" open={outcome.outcome === 'home'}>
      <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-sm font-medium hover:bg-secondary/40">
        <span>
          {OUTCOME_LABEL[outcome.outcome] ?? outcome.outcome} —{' '}
          <span className="tabular-nums">{formatPercent(outcome.model_probability)}</span>
        </span>
        <span className="text-xs text-muted-foreground tabular-nums">
          base {formatPercent(outcome.base_probability)}
        </span>
      </summary>
      <div className="px-4 pb-3 space-y-1 text-xs">
        {all.length === 0 ? (
          <div className="italic text-muted-foreground">no contributors (baseline prediction)</div>
        ) : (
          all.map((c) => {
            const width = (Math.abs(c.shap_value) / max) * 100
            const positive = c.sign === 'pos'
            return (
              <div key={`${c.feature}-${c.sign}`} className="flex items-center gap-2">
                <div className="w-40 truncate" title={`${c.feature} = ${c.value.toFixed(3)}`}>
                  {c.feature}
                </div>
                <div className="flex-1 h-2 bg-secondary rounded relative overflow-hidden">
                  <div
                    className={positive ? 'bg-[hsl(var(--success))]' : 'bg-destructive'}
                    style={{
                      width: `${width}%`,
                      height: '100%',
                      marginLeft: positive ? '50%' : `${50 - width}%`,
                    }}
                  />
                </div>
                <div className="w-16 text-right tabular-nums text-muted-foreground">
                  {positive ? '+' : ''}
                  {c.shap_value.toFixed(3)}
                </div>
              </div>
            )
          })
        )}
      </div>
    </details>
  )
}
