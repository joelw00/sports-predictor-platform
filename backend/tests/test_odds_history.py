from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db import models as m
from app.db.base import Base
from app.odds.history import mark_closing_odds


@pytest.fixture()
def db() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    s = TestSession()
    try:
        yield s
    finally:
        s.close()


def _seed_match(db: Session, kickoff: datetime) -> m.Match:
    sport = m.Sport(code="football", name="Football")
    db.add(sport)
    db.flush()
    comp = m.Competition(sport_id=sport.id, code="PL", name="Premier League")
    db.add(comp)
    db.flush()
    home = m.Team(competition_id=comp.id, name="Manchester City")
    away = m.Team(competition_id=comp.id, name="Arsenal")
    db.add_all([home, away])
    db.flush()
    match = m.Match(
        sport_id=sport.id,
        competition_id=comp.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=kickoff,
    )
    db.add(match)
    db.flush()
    return match


def test_mark_closing_picks_latest_pre_kickoff_snapshot(db: Session):
    kickoff = datetime(2025, 1, 15, 15, 0, tzinfo=UTC)
    match = _seed_match(db, kickoff)

    # Three snapshots at different times, same (book, market, selection, line).
    for minutes_before, price in [(120, 1.80), (30, 1.85), (5, 1.90)]:
        db.add(
            m.OddsSnapshot(
                match_id=match.id,
                bookmaker="snai",
                market="1X2",
                selection="home",
                price=price,
                captured_at=kickoff - timedelta(minutes=minutes_before),
            )
        )
    # One snapshot AFTER kickoff that should never be flagged.
    db.add(
        m.OddsSnapshot(
            match_id=match.id,
            bookmaker="snai",
            market="1X2",
            selection="home",
            price=1.95,
            captured_at=kickoff + timedelta(minutes=5),
        )
    )
    db.flush()

    marked = mark_closing_odds(db, now=kickoff + timedelta(minutes=10))
    assert marked == 1
    closing = (
        db.query(m.OddsSnapshot)
        .filter(m.OddsSnapshot.is_closing.is_(True))
        .one()
    )
    # Latest pre-kickoff snapshot wins (5 minutes before kickoff → price 1.90).
    assert closing.price == 1.90


def test_mark_closing_is_idempotent_and_resets_stale_flags(db: Session):
    kickoff = datetime(2025, 1, 15, 15, 0, tzinfo=UTC)
    match = _seed_match(db, kickoff)

    early = m.OddsSnapshot(
        match_id=match.id,
        bookmaker="snai",
        market="1X2",
        selection="home",
        price=1.80,
        captured_at=kickoff - timedelta(minutes=60),
        is_closing=True,  # stale flag from an earlier run
    )
    late = m.OddsSnapshot(
        match_id=match.id,
        bookmaker="snai",
        market="1X2",
        selection="home",
        price=1.85,
        captured_at=kickoff - timedelta(minutes=5),
    )
    db.add_all([early, late])
    db.flush()

    mark_closing_odds(db, now=kickoff + timedelta(minutes=10))
    assert late.is_closing is True
    assert early.is_closing is False

    # Second pass: same inputs, still one closing row on the latest snapshot.
    mark_closing_odds(db, now=kickoff + timedelta(minutes=20))
    assert late.is_closing is True
    assert early.is_closing is False


def test_mark_closing_skips_matches_that_have_not_started(db: Session):
    future_kickoff = datetime.now(tz=UTC) + timedelta(days=1)
    match = _seed_match(db, future_kickoff)
    db.add(
        m.OddsSnapshot(
            match_id=match.id,
            bookmaker="snai",
            market="1X2",
            selection="home",
            price=1.80,
            captured_at=datetime.now(tz=UTC),
        )
    )
    db.flush()

    marked = mark_closing_odds(db)
    assert marked == 0
    snap = db.query(m.OddsSnapshot).one()
    assert snap.is_closing is False
