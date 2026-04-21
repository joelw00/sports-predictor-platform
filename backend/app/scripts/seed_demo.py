"""Seed the database with demo data.

Usage:
    python -m app.scripts.seed_demo [--if-empty]
"""

from __future__ import annotations

import argparse

from app.db import SessionLocal
from app.db import models as m
from app.ingestion.demo import DemoSource
from app.ingestion.orchestrator import ingest_all
from app.logging import configure_logging, get_logger

log = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-empty", action="store_true")
    args = parser.parse_args()

    with SessionLocal() as db:
        if args.if_empty and db.query(m.Match).count() > 0:
            log.info("seed_demo.skip", reason="db already has matches")
            return
        counts = ingest_all(db, sources=[DemoSource()])
        log.info("seed_demo.done", **counts)


if __name__ == "__main__":
    main()
