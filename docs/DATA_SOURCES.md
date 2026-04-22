# Data sources

The platform is built around a **pluggable adapter interface** so new sources can be added without touching the ML or UI layers. Every adapter implements `backend/app/ingestion/base.py::BaseSource`.

A fully functional **demo adapter** (`ingestion/demo.py`) generates synthetic but realistic data so the whole system runs end-to-end without any external API. It is always registered last as a fallback.

The sections below document the external sources the platform is designed to integrate with. Actual adapter implementations land in **Phase 1** (see `docs/ROADMAP.md`); the stubs and contracts live in the repo today.

---

## Football

### API-Football (RapidAPI)

- **Docs**: https://www.api-football.com/documentation-v3
- **Access**: RapidAPI key. Free tier (100 req/day) is enough for a handful of leagues; paid tiers unlock full coverage.
- **Covers**: fixtures, results, statistics, lineups, standings, injuries, predictions, odds (limited).
- **Env var**: `API_FOOTBALL_KEY`.

### Football-Data.org

- **Docs**: https://www.football-data.org/documentation/api
- **Access**: free key with ~10 competitions included; paid tier for the rest.
- **Covers**: fixtures, scorers, standings, teams. Excellent reliability, low rate limits.
- **Adapter**: `backend/app/ingestion/football_data_org.py`. **Live as of PR 3.**
- **Env vars**:
  - `FOOTBALL_DATA_KEY` — API token (required to enable).
  - `FOOTBALL_DATA_COMPETITIONS` — comma-separated competition codes. Default:
    `PL,SA,BL1,FL1,PD,DED,PPL`.
  - `FOOTBALL_DATA_SEASON` — starting year, e.g. `2024` for 2024/25. Empty = current.
- **Rate limit**: 10 req/min on the free tier. The adapter spreads calls to
  roughly one every 6.5 s and retries 429/5xx with exponential backoff.
- **Raw audit**: every response is persisted into the `ingestion_payloads`
  table so we can rebuild `matches` without re-calling the API.
- **CLI**: `python -m app.scripts.ingest_football_data [--season 2024] [--competitions PL,SA]`.

### SofaScore (unofficial)

- **Access**: no official public API. The adapter uses best-effort public endpoints and honours `robots.txt` and sensible rate limits.
- **Covers**: very rich stats incl. player heatmaps, shot maps, live events.
- **Env var**: `SOFASCORE_ENABLED=true`. Disabled by default.
- **Note**: terms may change; the adapter is opt-in and easy to replace with Sportmonks or similar.

### Understat (scraping)

- **Access**: public pages on understat.com. The adapter parses the embedded JSON.
- **Covers**: xG/xA for top-5 European leagues and RPL.
- **Env var**: `UNDERSTAT_ENABLED=true`.

### The Odds API

- **Docs**: https://the-odds-api.com/
- **Access**: free tier (500 req/month) up to paid plans.
- **Covers**: odds from dozens of bookmakers across many sports including football and table tennis.
- **Env var**: `THE_ODDS_API_KEY`.

### SNAI (Italian market)

- **Access**: public website scraping. Respect robots.txt and throttle aggressively.
- **Covers**: pre-match and live odds for Italian market.
- **Env var**: `SNAI_ENABLED=true`.

---

## Table tennis

### ITTF / WTT

- **Access**: public pages and RSS feeds. Limited structured API.
- **Covers**: world ranking, results, schedules, singles & doubles.
- **Adapter**: lands in Phase 4.

### The Odds API (table tennis)

- Same key as above; `sport_key = table_tennis_*`.

---

## Weather enrichment

- **Open-Meteo**: https://open-meteo.com/ — free, no key required. Queried with venue lat/lon when available. Optional.

---

## Data quality and normalisation

- **Alias table** in `reference.aliases` maps source-specific names to canonical IDs.
- **Source priority** resolves conflicts when multiple sources disagree (configurable per entity).
- **Freshness checks** (rows per source per day) run after each ingestion job.
- **Audit trail**: `ingest_jobs` row per run with counts, duration, errors.

## Rate limiting & retries

- Token bucket per source.
- Exponential backoff + jitter on 5xx and network errors.
- Circuit breaker opens after N consecutive failures and cools down before retrying.

## Legal / ToS

Use only sources whose terms you can comply with. Scraping adapters are **off by default**. Commercial use typically requires a paid plan — check each provider's terms before enabling.
