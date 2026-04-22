from __future__ import annotations

from datetime import UTC

from app.db import models as m
from app.ingestion.demo import DemoSource
from app.ingestion.orchestrator import ingest_all


def test_demo_ingestion_populates_tables(db_session):
    counts = ingest_all(db_session, sources=[DemoSource(seed=7)])
    assert counts["matches"] > 0
    assert counts["odds"] > 0
    # At least one sport was created.
    assert db_session.query(m.Sport).count() >= 1
    # Some matches should be in the future.
    from datetime import datetime

    now = datetime.now(tz=UTC)
    upcoming = db_session.query(m.Match).filter(m.Match.kickoff >= now).count()
    assert upcoming > 0
