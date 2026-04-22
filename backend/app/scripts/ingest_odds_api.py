"""Run a one-shot ingestion from The Odds API.

Usage::

    python -m app.scripts.ingest_odds_api [--sport-keys soccer_epl,soccer_italy_serie_a]
                                          [--regions eu,uk]
                                          [--markets h2h,totals,spreads]

Defaults come from :class:`app.config.Settings` so the same flag set works in
CLI, scheduled job and tests.
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.db import SessionLocal
from app.ingestion.orchestrator import ingest_all
from app.ingestion.the_odds_api import TheOddsApiRealSource
from app.logging import configure_logging, get_logger

log = get_logger(__name__)


def main() -> None:
    configure_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sport-keys",
        default=",".join(settings.the_odds_api_sport_key_list),
        help="Comma-separated The Odds API sport keys.",
    )
    parser.add_argument(
        "--regions",
        default=",".join(settings.the_odds_api_region_list),
        help="Comma-separated bookmaker regions (eu,uk,us,au).",
    )
    parser.add_argument(
        "--markets",
        default=",".join(settings.the_odds_api_market_list),
        help="Comma-separated markets (h2h,totals,spreads).",
    )
    args = parser.parse_args()

    if not settings.the_odds_api_key:
        raise SystemExit(
            "THE_ODDS_API_KEY is not set — fill it in .env and retry, or run "
            "`python -m app.scripts.seed_demo` to use the demo adapter."
        )

    source = TheOddsApiRealSource(
        settings.the_odds_api_key,
        sport_keys=[s.strip() for s in args.sport_keys.split(",") if s.strip()],
        regions=[r.strip() for r in args.regions.split(",") if r.strip()],
        markets=[m.strip() for m in args.markets.split(",") if m.strip()],
    )
    try:
        with SessionLocal() as db:
            counts = ingest_all(db, sources=[source], trigger="cli")
            log.info("ingest_odds_api.done", **counts)
    finally:
        source.close()


if __name__ == "__main__":
    main()
