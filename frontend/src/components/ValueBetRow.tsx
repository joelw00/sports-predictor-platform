import { Link } from 'react-router-dom'
import { format } from 'date-fns'
import { Badge } from '@/components/ui/badge'
import { ValueBetOut } from '@/lib/api'
import { formatEdge, formatOdds, formatPercent, marketLabel } from '@/lib/utils'

export default function ValueBetRow({ bet }: { bet: ValueBetOut }) {
  return (
    <Link
      to={`/matches/${bet.match_id}`}
      className="grid grid-cols-12 items-center gap-3 rounded-md border border-border bg-card px-4 py-3 text-sm hover:border-primary/60 transition-colors"
    >
      <div className="col-span-3 truncate">
        <div className="font-medium truncate">
          {bet.home_team} <span className="text-muted-foreground">vs</span> {bet.away_team}
        </div>
        <div className="text-xs text-muted-foreground truncate">{bet.competition ?? '—'}</div>
      </div>
      <div className="col-span-2 text-xs text-muted-foreground">
        {bet.kickoff ? format(new Date(bet.kickoff), 'EEE d MMM · HH:mm') : '—'}
      </div>
      <div className="col-span-2">
        <Badge variant="outline">{marketLabel(bet.market, bet.selection, bet.line)}</Badge>
      </div>
      <div className="col-span-1 text-right tabular-nums">{formatOdds(bet.price)}</div>
      <div className="col-span-1 text-right tabular-nums">{formatPercent(bet.p_model)}</div>
      <div className="col-span-1 text-right tabular-nums">{formatPercent(bet.p_fair)}</div>
      <div className="col-span-1 text-right">
        <span className={bet.edge > 0 ? 'text-[hsl(var(--success))] font-semibold' : 'text-destructive'}>
          {formatEdge(bet.edge)}
        </span>
      </div>
      <div className="col-span-1 text-right text-xs text-muted-foreground">
        K {bet.kelly_fraction.toFixed(3)}
      </div>
    </Link>
  )
}
