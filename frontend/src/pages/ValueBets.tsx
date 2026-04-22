import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import ValueBetRow from '@/components/ValueBetRow'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

export default function ValueBetsPage() {
  const [sport, setSport] = useState('football')
  const [market, setMarket] = useState('all')
  const [minEdge, setMinEdge] = useState('0.03')
  const [minConf, setMinConf] = useState('0.55')

  const { data, isLoading } = useQuery({
    queryKey: ['value-bets', sport, market, minEdge, minConf],
    queryFn: () =>
      api.valueBets({
        sport,
        market: market === 'all' ? undefined : market,
        min_edge: Number(minEdge || 0),
        min_confidence: Number(minConf || 0),
        limit: 200,
      }),
    refetchInterval: 60_000,
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Value bet ranking</h1>
        <p className="text-sm text-muted-foreground">
          Opportunities where the model&apos;s fair probability exceeds the bookmaker&apos;s devigged implied probability.
        </p>
      </div>
      <Card>
        <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4">
          <div>
            <label className="text-xs text-muted-foreground">Sport</label>
            <Select value={sport} onValueChange={setSport}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="football">Football</SelectItem>
                <SelectItem value="table_tennis">Table Tennis</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Market</label>
            <Select value={market} onValueChange={setMarket}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="1x2">1X2</SelectItem>
                <SelectItem value="double_chance">Double chance</SelectItem>
                <SelectItem value="over_under">Over / Under</SelectItem>
                <SelectItem value="btts">BTTS</SelectItem>
                <SelectItem value="match_winner">Match winner (TT)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Min edge</label>
            <Input type="number" step="0.005" value={minEdge} onChange={(e) => setMinEdge(e.target.value)} />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Min confidence</label>
            <Input type="number" step="0.05" value={minConf} onChange={(e) => setMinConf(e.target.value)} />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-0">
          <div className="grid grid-cols-12 gap-3 px-4 py-2 text-xs uppercase tracking-wide text-muted-foreground border-b border-border">
            <div className="col-span-3">Match</div>
            <div className="col-span-2">Kickoff</div>
            <div className="col-span-2">Market</div>
            <div className="col-span-1 text-right">Odds</div>
            <div className="col-span-1 text-right">P(model)</div>
            <div className="col-span-1 text-right">P(fair)</div>
            <div className="col-span-1 text-right">Edge</div>
            <div className="col-span-1 text-right">Kelly</div>
          </div>
          <div className="flex flex-col divide-y divide-border">
            {(data ?? []).map((bet, idx) => (
              <ValueBetRow key={`${bet.match_id}-${bet.market}-${bet.selection}-${idx}`} bet={bet} />
            ))}
            {isLoading && (
              <div className="px-4 py-6 text-center text-sm text-muted-foreground">Loading value bets…</div>
            )}
            {!isLoading && (data ?? []).length === 0 && (
              <div className="px-4 py-6 text-center text-sm text-muted-foreground">
                No value bets meet the current thresholds.
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
