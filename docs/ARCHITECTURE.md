# Architecture

This document describes the end-to-end architecture of the Sports Predictor Platform. It is written to be read top-down: start with the system overview, then drill into layers.

## 1. System overview

```
                ┌─────────────────────────────────────────────────────────┐
                │                     Data sources                        │
                │  API-Football · Football-Data · SofaScore · Understat   │
                │  The Odds API · SNAI · (ITTF / WTT for table tennis)    │
                │  Demo adapter (synthetic data, always available)        │
                └────────────────────────┬────────────────────────────────┘
                                         │
                                 ┌───────▼───────┐
                                 │  Ingestion    │  pull + stream, retry, cache
                                 │  adapters     │
                                 └───────┬───────┘
                                         │
           ┌─────────────────────────────▼─────────────────────────────┐
           │                    Storage layer                          │
           │  PostgreSQL (relational) · Redis (cache + live bus)       │
           │  raw / clean / feature tables, odds, predictions, audit   │
           └───────────┬───────────────────────┬───────────────────────┘
                       │                       │
            ┌──────────▼──────────┐   ┌────────▼──────────┐
            │ Feature engineering │   │ Live event bus    │
            │ (batch + on-demand) │   │ (Redis pub/sub)   │
            └──────────┬──────────┘   └────────┬──────────┘
                       │                       │
            ┌──────────▼──────────┐   ┌────────▼──────────┐
            │ ML pipeline         │   │ Live prediction   │
            │ train · calibrate · │   │ service           │
            │ ensemble · evaluate │   │                   │
            └──────────┬──────────┘   └────────┬──────────┘
                       │                       │
           ┌───────────▼───────────────────────▼──────────┐
           │ Prediction service                           │
           │ produces calibrated probabilities per market │
           └───────────┬──────────────────────────────────┘
                       │
           ┌───────────▼──────────┐
           │ Value bet engine     │  compares p_model vs. p_implied
           │ edge · EV · ranking  │  filters, risk checks, confidence
           └───────────┬──────────┘
                       │
           ┌───────────▼──────────┐
           │ FastAPI REST + WS    │
           └───────────┬──────────┘
                       │
           ┌───────────▼──────────┐
           │ React + Vite + shadcn│
           │ dashboard            │
           └──────────────────────┘
```

## 2. Layered responsibilities

### 2.1 Ingestion

Adapters live under `backend/app/ingestion/`. Each adapter implements the `BaseSource` protocol and returns normalised entities (matches, teams, players, odds snapshots). Shared concerns:

- **Rate limiting** — per-source token bucket.
- **Retries** — exponential backoff with jitter on 5xx / network errors.
- **Caching** — Redis-backed, TTL per endpoint class.
- **Deduplication** — by `(source, external_id, captured_at)`.
- **Normalisation** — canonical team/player IDs (fuzzy match + manual alias table).
- **Mode flag** — each adapter exposes `is_enabled()` so the orchestrator can fall back to the demo adapter transparently.

### 2.2 Cleaning

Dedicated `cleaning` routines (invoked from the ingestion DAG) enforce:

- type coercion and unit normalisation (timezones → UTC),
- missing-value policies per column,
- cross-source reconciliation (prefer `source_priority` ordering),
- alias resolution for teams/players/leagues,
- outlier detection on numeric fields.

### 2.3 Storage

PostgreSQL schema groups (see `backend/alembic/`):

- **reference** — `sports`, `competitions`, `seasons`, `teams`, `players`, `venues`, `aliases`.
- **events** — `matches`, `match_stats`, `lineups`, `events_live`.
- **odds** — `odds_snapshots`, `markets`, `selections`, `closing_lines`.
- **features** — `match_features_football`, `match_features_tt`, feature store versions.
- **ml** — `model_registry`, `predictions`, `prediction_markets`, `calibration_runs`.
- **analytics** — `value_bets`, `backtest_runs`, `backtest_results`.
- **audit** — `ingest_jobs`, `pipeline_runs`, `data_quality_checks`.

Redis holds:

- live score bus (`pubsub` channel per match),
- latest-odds cache,
- feature/prediction cache for hot reads,
- rate-limit counters.

Large blobs (raw HTML/JSON responses, trained model artifacts) go to local `data/` and `models/artifacts/` directories. Swap these for S3/GCS in production via the `StorageBackend` abstraction.

### 2.4 Feature engineering

Features are computed by idempotent, versioned functions. For football:

- Elo rating with home advantage,
- form (last N matches, weighted),
- rolling xG for/against,
- shots / SoT / corners / cards rolling averages,
- BTTS / Over rates,
- rest days, travel distance, fixture congestion,
- head-to-head aggregates,
- style of play vectors,
- lineup strength (from player ratings when available),
- match importance score,
- derby / title-race / relegation flags,
- weather (temperature, precipitation, wind) when venue coordinates known,
- referee tendencies.

For table tennis:

- Elo,
- rolling win rate (overall, last N, surface/format),
- set win rate, first-set win rate,
- points per set, avg match duration,
- serve/receive efficiency when available,
- head-to-head,
- tournament level weighting.

Each feature function is a pure transform with a version string recorded alongside the output row, enabling reproducibility.

### 2.5 ML pipeline

Three classes of models coexist:

1. **Statistical baselines** — Poisson double for football goals (independent + bivariate Dixon–Coles); Elo-based logistic for match winner.
2. **Gradient-boosted models** — LightGBM / XGBoost per target (1X2, O/U lines, BTTS, corners, cards, scoreline distribution). Isotonic calibration on out-of-fold predictions.
3. **Ensembles** — weighted averaging of calibrated probabilities, weights learned on a held-out window (`scipy.optimize`).

Training runs are orchestrated by `app.ml.pipeline.Trainer`:

- walk-forward CV split by match date,
- Optuna hyperparameter search (time-budgeted),
- calibration fit on the last fold,
- artifact persisted with feature list + version + metrics,
- registry entry in `model_registry`.

Prediction-time inference reads the active model per `(sport, market)` from the registry.

### 2.6 Live prediction

- Ingestion writes live events (`goal`, `card`, `corner`, `sub`, `point`, `set_end`) onto Redis.
- `LivePredictor` maintains per-match state and triggers re-scoring on each relevant event.
- For football: piecewise linear decay of pre-match probabilities combined with an in-play goal-rate model conditioned on current score, minute, red cards.
- For table tennis: point-by-point win probability via Markov chain over the current set/match state.
- WebSocket endpoint `/ws/live/{match_id}` streams updated probabilities to the frontend.

### 2.7 Value bet engine

For each `(match, market, selection)`:

```
p_model    = calibrated model probability
p_implied  = 1 / odds
margin     = sum(p_implied over market) − 1
p_fair     = p_implied / (1 + margin)          # bookmaker overround removed
edge       = p_model − p_fair
ev         = p_model * (odds − 1) − (1 − p_model)
kelly      = max(0, edge / (odds − 1))
```

Selection rules (all configurable via env):

- `edge ≥ VALUE_BET_MIN_EDGE`,
- model `confidence ≥ VALUE_BET_MIN_CONFIDENCE` (lower bound of bootstrap CI),
- sufficient data volume for the (team, market) pair,
- market not flagged high-variance,
- no strong contrary odds movement (CLV sanity check).

Output ranked by `edge`, `ev`, and a composite stability score.

### 2.8 Backtesting

`app.backtesting.engine.Backtester`:

- replays a historical window,
- uses the **opening** and **closing** odds as stored,
- simulates flat-stake and Kelly-fraction strategies,
- computes ROI, yield, profit factor, max drawdown, hit rate, CLV,
- breaks results down by sport, league, market, edge bucket,
- persists to `backtest_runs` / `backtest_results` for the UI.

### 2.9 API

FastAPI routers under `backend/app/api/`:

- `GET /health`
- `GET /sports`
- `GET /events?sport&date&competition&status` — today's / upcoming matches with predictions
- `GET /events/{id}` — full detail incl. stats, lineups, predictions, odds history
- `GET /predictions?sport&market&date`
- `GET /value-bets?sport&min_edge&min_confidence&date`
- `GET /backtests` / `POST /backtests/run`
- `WS  /ws/live/{match_id}`

All responses are Pydantic-validated. OpenAPI at `/docs`.

### 2.10 Frontend

React 18 + Vite + TypeScript + TailwindCSS + shadcn/ui + Radix primitives. State via TanStack Query (server cache) and Zustand (UI). Charts via Recharts. Dark mode by default.

Pages:

- **Dashboard** — today's events, quick filters, highlights.
- **Value Bets** — ranked table with edge/EV/confidence, sport and market filters.
- **Match Detail** — stats, probabilities, odds history chart, explanation panel.
- **Backtest** — run list, equity curve, breakdown charts.
- **About** — methodology + disclaimer.

### 2.11 Observability

- Structured JSON logs via `structlog`.
- `/metrics` Prometheus endpoint (FastAPI instrumentator).
- Model drift: population stability index on feature distributions vs. training window; alert when PSI > 0.25.
- Prediction drift: Brier score rolling window.
- Data quality: row-count, null-rate, and freshness checks per source.

## 3. Scaling considerations

- **Horizontal API** — stateless FastAPI behind a load balancer.
- **Workers** — ingestion and training run as background workers (RQ / Celery). The compose setup inlines them into the API container for simplicity; split them out in production.
- **Database** — read replicas for analytics queries; partition `odds_snapshots` by month.
- **Caching** — aggressive Redis caching of `/events` and `/value-bets` with short TTL + pub/sub invalidation on live updates.
- **Object storage** — move `data/` and `models/artifacts/` to S3/GCS behind the `StorageBackend` interface.

## 4. Security & compliance

- Secrets only via environment / secret manager; never committed.
- Rate-limited public API, CORS restricted to known frontends.
- PII-free by design (no user accounts in v1; add OAuth later).
- Respect each data source's ToS and robots.txt; scraping adapters are opt-in.
- Clear "not a guarantee" disclaimer in UI and README.

## 5. Non-goals (v1)

- Placing bets automatically.
- Handling user bankrolls or real money.
- Markets that require proprietary data the platform cannot access.

## 6. Directory map

```
backend/app/
├── ingestion/       # adapters + orchestrator
├── features/        # versioned feature transforms
├── ml/              # models, calibration, ensembles, explainer
├── predictions/     # inference service
├── value_bet/       # edge/EV engine
├── backtesting/     # historical simulator
├── live/            # websocket + in-play models
├── api/             # FastAPI routers
├── db/              # SQLAlchemy models
├── schemas/         # Pydantic I/O schemas
└── scripts/         # seed, train, maintenance CLIs

frontend/src/
├── components/      # shadcn primitives + composites
├── pages/           # Dashboard, ValueBets, MatchDetail, Backtest, About
├── lib/             # api client, utils
└── hooks/           # data hooks (TanStack Query)
```
