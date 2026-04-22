"""Run a one-shot ingestion from Football-Data.org.

Usage::

    python -m app.scripts.ingest_football_data [--season 2024] [--competitions PL,SA,BL1]

Flags pull from :class:`app.config.Settings` by default so the same defaults
work in CLI, scheduled job and tests.
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.db import SessionLocal
from app.ingestion.football_data_org import FootballDataOrgSource
from app.ingestion.orchestrator import ingest_all
from app.logging import configure_logging, get_logger

log = get_logger(__name__)


def main() -> None:
    configure_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        default=settings.football_data_season or None,
        help="Starting year of the season, e.g. '2024' (default: current season).",
    )
    parser.add_argument(
        "--competitions",
        default=",".join(settings.football_data_competition_list),
        help="Comma-separated Football-Data.org competition codes.",
    )
    args = parser.parse_args()

    if not settings.football_data_key:
        raise SystemExit(
            "FOOTBALL_DATA_KEY is not set — fill it in .env and retry, or run "
            "`python -m app.scripts.seed_demo` to use the demo adapter."
        )

    source = FootballDataOrgSource(
        settings.football_data_key,
        competitions=[c.strip() for c in args.competitions.split(",") if c.strip()],
        season=args.season,
    )
    try:
        with SessionLocal() as db:
            counts = ingest_all(db, sources=[source], trigger="cli")
            log.info("ingest_football_data.done", **counts)
    finally:
        source.close()


if __name__ == "__main__":
    main()
