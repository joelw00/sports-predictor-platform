# Roadmap

The platform is delivered in incremental phases. Each phase ships as a separate PR and keeps the system runnable end-to-end.

## Phase 0 — Foundation (this PR)

- [x] Monorepo scaffolding with `backend/`, `frontend/`, `docs/`, Docker Compose, CI.
- [x] PostgreSQL schema (SQLAlchemy + Alembic) covering sports, teams, players, matches, odds, predictions, value bets, backtests.
- [x] Demo data adapter generating realistic football fixtures, match stats and odds so the full stack runs out of the box.
- [x] Feature engineering (Elo, rolling form, goal rates, H2H, rest days) for football.
- [x] Baseline ML: independent Poisson goal model + LightGBM 1X2/OU with isotonic calibration.
- [x] Value bet engine with edge, EV, Kelly, configurable thresholds and ranking.
- [x] Backtesting engine (flat stake, Kelly fraction) with ROI, yield, drawdown, profit factor.
- [x] FastAPI: events, predictions, value bets, backtests, health.
- [x] React + Vite + Tailwind + shadcn/ui dashboard: today's events, value bet ranking, match detail, backtest view, about.
- [x] Pytest + ruff + mypy + frontend lint/build in CI.
- [x] Table tennis domain modelled in the schema and demo data (baseline models in Phase 4).

## Phase 1 — Real data sources

- [ ] API-Football adapter (RapidAPI).
- [ ] Football-Data.org adapter.
- [ ] The Odds API adapter (for odds).
- [ ] SofaScore adapter (unofficial).
- [ ] Understat adapter (xG, scraping).
- [ ] SNAI adapter (public scraping of Italian market).
- [ ] Alias tables + fuzzy matching across sources.
- [ ] Scheduled ingestion jobs (APScheduler / cron) with run history UI.

## Phase 2 — Feature engineering depth

- [ ] Lineup-aware strength features (player ratings, injuries, suspensions).
- [ ] xG rolling model with opponent adjustment.
- [ ] Match importance scoring (relegation, title race, derby, cup rounds).
- [ ] Referee tendencies.
- [ ] Weather enrichment.
- [ ] Feature store versioning UI.

## Phase 3 — Advanced football markets

- [ ] Dixon–Coles correction for low-score dependence.
- [ ] Asian / European handicap pricing from the scoreline distribution.
- [ ] HT/FT, correct score, clean sheet, comeback probability.
- [ ] Corners and cards models (Poisson / NB).
- [ ] Goalscorer probabilities (shot-based, minute-distributed).
- [ ] First/second half splits.
- [ ] Combo bet pricer with correlation adjustment.

## Phase 4 — Table tennis

- [ ] ITTF / WTT ingestion adapter.
- [ ] Elo + form + H2H features.
- [ ] Point-level Markov model for live win probability.
- [ ] Set winner, total points, handicap, exact set score markets.
- [ ] Dashboard tab for ping pong with ranking and live ticker.

## Phase 5 — Live prediction

- [ ] Event-driven live ingestion (football: goal / card / sub / corner).
- [ ] In-play football model (goal-rate conditioned on state).
- [ ] Point-by-point table tennis live updates.
- [ ] WebSocket streaming to the frontend; live badge and animated probability bars.

## Phase 6 — Explainability

- [ ] SHAP values per prediction cached at inference time.
- [ ] "Why this pick" panel in the UI with top drivers and risks.
- [ ] Bootstrap confidence intervals on probabilities.
- [ ] Model cards per `(sport, market)`.

## Phase 7 — Risk management & reporting

- [ ] Exposure limits (per league, market, correlation group).
- [ ] Automatic exclusion of high-variance events.
- [ ] Daily / weekly / monthly reports (PDF + email hook).
- [ ] Top events of the day feed in the dashboard.

## Phase 8 — Hardening & ops

- [ ] Structured logging pipeline, Prometheus + Grafana dashboards.
- [ ] Drift detectors with alerts.
- [ ] S3/GCS storage backend.
- [ ] OAuth auth + multi-user dashboards.
- [ ] Production deploy recipe (Terraform or similar).
