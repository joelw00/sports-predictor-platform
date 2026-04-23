import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { format } from 'date-fns'
import { ArrowLeft } from 'lucide-react'
import { api } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { formatEdge, formatOdds, formatPercent, marketGroup, marketLabel } from '@/lib/utils'

export default function MatchDetailPage() {
  const { id } = useParams()
  const matchId = Number(id)
  const { data, isLoading, error } = useQuery({
    queryKey: ['match', matchId],
    queryFn: () => api.matchDetail(matchId),
    enabled: Number.isFinite(matchId),
  })

  if (isLoading) return <div className="text-muted-foreground text-sm">Loading match…</div>
  if (error || !data) return <div className="text-destructive text-sm">Match not found.</div>

  const oddsByKey = new Map<string, typeof data.odds[number][]>()
  for (const o of data.odds) {
    const key = `${o.market}|${o.selection}|${o.line ?? ''}`
    const arr = oddsByKey.get(key) ?? []
    arr.push(o)
    oddsByKey.set(key, arr)
  }

  return (
    <div className="space-y-6">
      <Button variant="ghost" asChild size="sm"><Link to="/"><ArrowLeft className="mr-1 h-4 w-4"/> back</Link></Button>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>
                {data.home_team.name} <span className="text-muted-foreground font-normal">vs</span> {data.away_team.name}
              </CardTitle>
              <div className="text-sm text-muted-foreground mt-1">
                {data.competition ?? '—'} · {format(new Date(data.kickoff), 'EEE d MMM yyyy · HH:mm')}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="uppercase">{data.sport.replace('_', ' ')}</Badge>
              <Badge variant={data.status === 'finished' ? 'secondary' : 'default'}>{data.status}</Badge>
            </div>
          </div>
        </CardHeader>
        {data.status === 'finished' && (
          <CardContent className="pt-0 text-3xl font-semibold tabular-nums">
            {data.home_score} – {data.away_score}
          </CardContent>
        )}
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle>Model predictions</CardTitle></CardHeader>
          <CardContent className="p-0">
            {data.predictions.length === 0 ? (
              <div className="px-4 py-4 text-sm italic text-muted-foreground">
                no predictions generated yet
              </div>
            ) : (
              (() => {
                const groups = new Map<string, typeof data.predictions>()
                for (const p of data.predictions) {
                  const key = marketGroup(p.market)
                  const arr = groups.get(key) ?? []
                  arr.push(p)
                  groups.set(key, arr)
                }
                return (
                  <div className="divide-y divide-border">
                    {Array.from(groups.entries()).map(([group, preds]) => (
                      <details
                        key={group}
                        className="group"
                        open={group === 'Match result' || group === 'Goals (Over / Under)'}
                      >
                        <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-sm font-medium hover:bg-secondary/40">
                          <span>{group}</span>
                          <span className="text-xs text-muted-foreground">
                            {preds.length} {preds.length === 1 ? 'selection' : 'selections'}
                          </span>
                        </summary>
                        <table className="w-full text-sm">
                          <thead className="text-xs uppercase text-muted-foreground">
                            <tr>
                              <th className="text-left px-4 py-2">Selection</th>
                              <th className="text-right px-4 py-2">Probability</th>
                              <th className="text-right px-4 py-2">Confidence</th>
                            </tr>
                          </thead>
                          <tbody>
                            {preds.map((p, i) => (
                              <tr key={`${p.market}-${p.selection}-${p.line ?? ''}-${i}`} className="border-t border-border">
                                <td className="px-4 py-2">{marketLabel(p.market, p.selection, p.line)}</td>
                                <td className="px-4 py-2 text-right tabular-nums">
                                  {formatPercent(p.probability)}
                                </td>
                                <td className="px-4 py-2 text-right tabular-nums text-muted-foreground">
                                  {formatPercent(p.confidence)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </details>
                    ))}
                  </div>
                )
              })()
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Value bets</CardTitle></CardHeader>
          <CardContent className="p-0">
            {data.value_bets.length === 0 ? (
              <div className="p-4 text-sm text-muted-foreground italic">no positive edge detected</div>
            ) : (
              <table className="w-full text-sm">
                <thead className="text-xs uppercase text-muted-foreground">
                  <tr>
                    <th className="text-left px-4 py-2">Market</th>
                    <th className="text-left px-4 py-2">Book</th>
                    <th className="text-right px-4 py-2">Odds</th>
                    <th className="text-right px-4 py-2">Edge</th>
                  </tr>
                </thead>
                <tbody>
                  {data.value_bets.map((vb, i) => (
                    <tr key={i} className="border-t border-border">
                      <td className="px-4 py-2">{marketLabel(vb.market, vb.selection, vb.line)}</td>
                      <td className="px-4 py-2">{vb.bookmaker}</td>
                      <td className="px-4 py-2 text-right tabular-nums">{formatOdds(vb.price)}</td>
                      <td className={`px-4 py-2 text-right font-semibold ${vb.edge > 0 ? 'text-[hsl(var(--success))]' : 'text-destructive'}`}>
                        {formatEdge(vb.edge)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle>Bookmaker odds</CardTitle></CardHeader>
        <CardContent className="p-0 overflow-auto">
          <table className="w-full text-sm">
            <thead className="text-xs uppercase text-muted-foreground">
              <tr>
                <th className="text-left px-4 py-2">Market</th>
                <th className="text-left px-4 py-2">Selection</th>
                <th className="text-right px-4 py-2">Price</th>
                <th className="text-left px-4 py-2">Book</th>
                <th className="text-left px-4 py-2">Captured</th>
              </tr>
            </thead>
            <tbody>
              {data.odds.map((o, i) => (
                <tr key={i} className="border-t border-border">
                  <td className="px-4 py-2">{o.market}</td>
                  <td className="px-4 py-2">{marketLabel(o.market, o.selection, o.line)}</td>
                  <td className="px-4 py-2 text-right tabular-nums">{formatOdds(o.price)}</td>
                  <td className="px-4 py-2">{o.bookmaker}</td>
                  <td className="px-4 py-2 text-muted-foreground">
                    {format(new Date(o.captured_at), 'd MMM HH:mm')}
                  </td>
                </tr>
              ))}
              {data.odds.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-4 text-center text-muted-foreground italic">
                    no odds captured
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
