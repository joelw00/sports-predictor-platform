# Sports Predictor Platform

ML-powered predictive analytics platform for sports betting, with primary focus on **football** and extended support for **table tennis (ping pong)**.

The system ingests historical and live data, builds engineered features, trains calibrated probabilistic models, compares model probabilities against bookmaker odds, and surfaces **value bets** (positive-expected-value opportunities) through a modern web dashboard.

> **Disclaimer.** This project does not promise guaranteed winnings. It works with probabilities, calibration, expected value and risk. Use responsibly.

---

## Highlights

- **Modular FastAPI backend** split into clear domains: `ingestion`, `features`, `ml`, `predictions`, `value_bet`, `backtesting`, `live`, `api`.
- **Pluggable data sources** — adapters for API-Football, Football-Data.org, SofaScore, Understat, The Odds API, SNAI, plus a **demo source** that generates realistic synthetic data so the full stack runs end-to-end out of the box.
- **Calibrated probabilistic models** — Poisson goal model + gradient boosted ensemble with isotonic calibration for 1X2, Over/Under, BTTS.
- **Value bet engine** — model probability vs. implied probability, edge %, expected value, confidence, ranking.
- **Backtesting engine** with ROI, yield, drawdown, profit factor.
- **Modern dashboard** — React + Vite + TailwindCSS + shadcn/ui. Event list, filters, match detail, value bet ranking, backtest report.
- **Explainability** — feature importance per pick and human-readable rationale.
- **Ping pong support** — separate models for match winner, set winner, total points, handicap.
- **Docker Compose** one-command local dev: PostgreSQL + Redis + API + Frontend.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full system design, [`docs/ROADMAP.md`](docs/ROADMAP.md) for the delivery plan, [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for source integration notes, and [`docs/MODELS.md`](docs/MODELS.md) for the modelling approach.

---

## Quick start

### 1. Clone & start everything

```bash
git clone https://github.com/joelw00/sports-predictor-platform.git
cd sports-predictor-platform
cp .env.example .env
docker compose up --build
```

Services:

| Service   | URL                       | Description            |
|-----------|---------------------------|------------------------|
| API       | http://localhost:8000     | FastAPI backend        |
| API docs  | http://localhost:8000/docs| OpenAPI / Swagger UI   |
| Frontend  | http://localhost:5173     | React dashboard        |
| Postgres  | localhost:5432            | Relational store       |
| Redis     | localhost:6379            | Cache / live bus       |

### 2. Seed demo data and train the baseline model

```bash
docker compose exec api python -m app.scripts.seed_demo
docker compose exec api python -m app.scripts.train_baseline
```

### 3. Open the dashboard

Visit http://localhost:5173 — you will see today's events, predictions, and ranked value bets from the demo dataset.

---

## Running locally without Docker

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Testing and linting

```bash
# Backend
cd backend
pytest
ruff check .
mypy app

# Frontend
cd frontend
npm run lint
npm run build
```

CI runs these on every push — see `.github/workflows/ci.yml`.

---

## Providing real data sources

The platform ships with a **demo adapter** that fabricates realistic data so the pipeline runs end-to-end. To switch to real sources set the corresponding environment variables in `.env`:

| Env var                  | Source                |
|--------------------------|-----------------------|
| `API_FOOTBALL_KEY`       | RapidAPI API-Football |
| `FOOTBALL_DATA_KEY`      | football-data.org     |
| `SOFASCORE_ENABLED`      | SofaScore (unofficial)|
| `UNDERSTAT_ENABLED`      | Understat (scraping)  |
| `THE_ODDS_API_KEY`       | The Odds API          |
| `SNAI_ENABLED`           | SNAI (public scraping)|

See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for access details, rate limits and terms of service notes for each provider.

---

## Project layout

```
sports-predictor-platform/
├── backend/        FastAPI app, ML pipeline, DB models, tests
├── frontend/       React + Vite + Tailwind + shadcn dashboard
├── docs/           Architecture, data sources, models, roadmap
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## License

MIT — see [LICENSE](LICENSE).
