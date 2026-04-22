import { Link } from 'react-router-dom'
import { format } from 'date-fns'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Event } from '@/lib/api'
import { formatEdge, formatPercent, marketLabel } from '@/lib/utils'

export default function EventCard({ event }: { event: Event }) {
  const pred = event.top_prediction
  const value = event.best_value
  return (
    <Link to={`/matches/${event.id}`} className="block group">
      <Card className="transition-colors group-hover:border-primary/50">
        <CardContent className="flex flex-col gap-3 p-4">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="uppercase">
                {event.sport.replace('_', ' ')}
              </Badge>
              {event.competition && <span>{event.competition}</span>}
            </div>
            <span>{format(new Date(event.kickoff), 'EEE d MMM · HH:mm')}</span>
          </div>
          <div className="flex items-center justify-between gap-3">
            <div className="flex-1 truncate">
              <div className="font-medium">{event.home_team}</div>
              <div className="text-muted-foreground text-sm">vs</div>
              <div className="font-medium">{event.away_team}</div>
            </div>
            <div className="text-right">
              {event.status === 'finished' ? (
                <div className="text-2xl font-semibold tabular-nums">
                  {event.home_score} – {event.away_score}
                </div>
              ) : (
                <Badge variant="secondary">{event.status}</Badge>
              )}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <div className="text-muted-foreground text-xs">Top pick</div>
              {pred ? (
                <div className="font-medium">
                  {marketLabel(pred.market, pred.selection, pred.line)} ·{' '}
                  <span className="text-primary">{formatPercent(pred.probability)}</span>
                </div>
              ) : (
                <div className="text-muted-foreground italic">no prediction</div>
              )}
            </div>
            <div>
              <div className="text-muted-foreground text-xs">Best value</div>
              {value ? (
                <div className="font-medium">
                  {marketLabel(value.market, value.selection, value.line)} ·{' '}
                  <span className={value.edge > 0 ? 'text-[hsl(var(--success))]' : 'text-destructive'}>
                    {formatEdge(value.edge)}
                  </span>
                </div>
              ) : (
                <div className="text-muted-foreground italic">no value detected</div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}
