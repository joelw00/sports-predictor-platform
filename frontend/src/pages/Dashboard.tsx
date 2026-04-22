import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import EventCard from '@/components/EventCard'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'

export default function Dashboard() {
  const sport = 'football'
  const [query, setQuery] = useState('')

  const { data, isLoading, error } = useQuery({
    queryKey: ['events', sport],
    queryFn: () => api.events({ sport }),
    refetchInterval: 60_000,
  })

  const filtered = useMemo(() => {
    if (!data?.items) return []
    const q = query.trim().toLowerCase()
    if (!q) return data.items
    return data.items.filter(
      (e) =>
        e.home_team.toLowerCase().includes(q) ||
        e.away_team.toLowerCase().includes(q) ||
        (e.competition ?? '').toLowerCase().includes(q),
    )
  }, [data, query])

  const stats = useMemo(() => {
    const items = data?.items ?? []
    return {
      total: items.length,
      upcoming: items.filter((e) => e.status !== 'finished').length,
      withValue: items.filter((e) => e.best_value && e.best_value.edge > 0).length,
    }
  }, [data])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Today&apos;s board</h1>
        <p className="text-sm text-muted-foreground">
          Probabilistic forecasts across football leagues, ranked by expected value.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard label="Events in window" value={stats.total} />
        <StatCard label="Upcoming" value={stats.upcoming} />
        <StatCard label="With positive edge" value={stats.withValue} tone="success" />
      </div>

      <div className="flex flex-col md:flex-row md:items-center gap-3 md:justify-between">
        <div />
        <div className="flex gap-2">
            <Input
              placeholder="Filter teams or league…"
              className="md:w-[260px]"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          <Select defaultValue="all">
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="scheduled">Scheduled</SelectItem>
              <SelectItem value="finished">Finished</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {isLoading && <div className="text-muted-foreground text-sm">Loading events…</div>}
      {error && <div className="text-destructive text-sm">Failed to load events.</div>}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filtered.map((event) => (
          <EventCard key={event.id} event={event} />
        ))}
      </div>
      {!isLoading && filtered.length === 0 && (
        <Card>
          <CardContent className="p-6 text-center text-muted-foreground text-sm">
            No events match the current filters.
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function StatCard({
  label,
  value,
  tone,
}: {
  label: string
  value: number | string
  tone?: 'default' | 'success'
}) {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
        <div
          className={`text-3xl font-semibold tabular-nums mt-1 ${
            tone === 'success' ? 'text-[hsl(var(--success))]' : ''
          }`}
        >
          {value}
        </div>
      </CardContent>
    </Card>
  )
}
