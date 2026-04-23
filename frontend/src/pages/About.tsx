import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function About() {
  return (
    <div className="space-y-6 max-w-3xl">
      <Card>
        <CardHeader><CardTitle>About this platform</CardTitle></CardHeader>
        <CardContent className="prose prose-invert max-w-none text-sm leading-relaxed text-muted-foreground space-y-3">
          <p>
            Sports Predictor Platform is a modular ML system that ingests data from multiple sources
            (API-Football, Football-Data, SofaScore, Understat, The Odds API, SNAI), engineers features
            (Elo, rolling form, xG, head-to-head), fits calibrated probabilistic models (Poisson, gradient
            boosting, ensembles) and ranks value bets against bookmaker odds.
          </p>
          <p>
            Coverage targets all major football leagues (Serie A, Premier League, LaLiga, Bundesliga, Ligue 1,
            Eredivisie, Primeira Liga, MLS, Liga MX, Brasileirão, J-League, UEFA and CONMEBOL competitions…).
            When no external data source is configured, a deterministic <em>Demo mode</em> generates realistic
            synthetic fixtures so the whole pipeline runs end-to-end.
          </p>
          <p className="text-warning-foreground">
            <strong>Important:</strong> predictions are <em>probabilistic</em>, not guarantees. Always apply risk
            management and never bet more than you can afford to lose.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
