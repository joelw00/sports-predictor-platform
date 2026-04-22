const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`${resp.status}: ${text || resp.statusText}`)
  }
  return (await resp.json()) as T
}

export interface HealthResponse {
  status: string
  version: string
  demo_mode: boolean
  env: string
}

export interface Sport {
  id: number
  code: string
  name: string
}

export interface PredictionOut {
  market: string
  selection: string
  line: number | null
  probability: number
  confidence: number
  model_version: string
  drivers: Record<string, number>
}

export interface ValueBetOut {
  match_id: number
  market: string
  selection: string
  line: number | null
  bookmaker: string
  price: number
  p_model: number
  p_fair: number
  edge: number
  expected_value: number
  kelly_fraction: number
  confidence: number
  rationale: Record<string, number | string>
  home_team: string | null
  away_team: string | null
  competition: string | null
  kickoff: string | null
  sport: string | null
}

export interface Event {
  id: number
  sport: string
  competition: string | null
  home_team: string
  away_team: string
  kickoff: string
  status: string
  home_score: number | null
  away_score: number | null
  top_prediction: PredictionOut | null
  best_value: ValueBetOut | null
}

export interface EventList {
  items: Event[]
  total: number
}

export interface OddsQuote {
  bookmaker: string
  market: string
  selection: string
  line: number | null
  price: number
  captured_at: string
  is_closing: boolean
  is_live: boolean
}

export interface MatchDetail {
  id: number
  sport: string
  competition: string | null
  home_team: { id: number; name: string }
  away_team: { id: number; name: string }
  kickoff: string
  status: string
  home_score: number | null
  away_score: number | null
  predictions: PredictionOut[]
  odds: OddsQuote[]
  value_bets: ValueBetOut[]
  form: Record<string, unknown>
}

export interface BacktestResult {
  id: number
  label: string
  sport_code: string
  market: string
  start_date: string
  end_date: string
  strategy: string
  min_edge: number
  stake: number
  n_bets: number
  n_wins: number
  total_staked: number
  total_return: number
  roi: number
  yield_pct: number
  max_drawdown: number
  profit_factor: number
  equity_curve: Array<{ date: string; bankroll: number; edge: number }>
  breakdown: Record<string, unknown>
}

export const api = {
  health: () => http<HealthResponse>('/health'),
  sports: () => http<Sport[]>('/sports'),
  events: (params: Record<string, string | undefined> = {}) => {
    const qs = new URLSearchParams()
    for (const [k, v] of Object.entries(params)) if (v) qs.set(k, v)
    return http<EventList>(`/events${qs.toString() ? `?${qs.toString()}` : ''}`)
  },
  matchDetail: (id: number) => http<MatchDetail>(`/events/${id}`),
  valueBets: (params: Record<string, string | number | undefined> = {}) => {
    const qs = new URLSearchParams()
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== '') qs.set(k, String(v))
    return http<ValueBetOut[]>(`/value-bets${qs.toString() ? `?${qs.toString()}` : ''}`)
  },
  backtests: () => http<BacktestResult[]>('/backtests'),
  runBacktest: (body: Partial<BacktestResult> & Record<string, unknown>) =>
    http<BacktestResult>('/backtests/run', { method: 'POST', body: JSON.stringify(body) }),
}
