"""Unit tests for :func:`app.monitoring.performance.compute_live_performance`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.db import models as m
from app.monitoring.performance import compute_live_performance


def _seed_team(db: Session, sport: m.Sport, name: str) -> m.Team:
    team = m.Team(name=name)
    db.add(team)
    db.flush()
    return team


def _make_match(
    db: Session,
    sport: m.Sport,
    home: m.Team,
    away: m.Team,
    *,
    kickoff: datetime,
    hs: int,
    as_: int,
) -> m.Match:
    match = m.Match(
        sport_id=sport.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=kickoff,
        status="finished",
        home_score=hs,
        away_score=as_,
    )
    db.add(match)
    db.flush()
    return match


def _add_1x2_preds(db: Session, match: m.Match, probs: tuple[float, float, float]) -> None:
    for sel, p in zip(("home", "draw", "away"), probs, strict=True):
        db.add(
            m.Prediction(
                match_id=match.id,
                market="1x2",
                selection=sel,
                line=None,
                probability=p,
                confidence=0.6,
                model_version="v-test",
                drivers={},
            )
        )
    db.flush()


def test_empty_when_no_finished_matches(db_session: Session) -> None:
    sport = m.Sport(code="football", name="Football")
    db_session.add(sport)
    db_session.commit()
    res = compute_live_performance(db_session, market="1x2", window_days=7)
    assert res.n == 0
    assert res.brier is None


def test_1x2_brier_matches_hand_computed(db_session: Session) -> None:
    now = datetime.now(UTC)
    sport = m.Sport(code="football", name="Football")
    db_session.add(sport)
    db_session.commit()
    db_session.refresh(sport)
    home = _seed_team(db_session, sport, "HomeFC")
    away = _seed_team(db_session, sport, "AwayFC")

    # Two matches, both decided 2-0 (home wins).
    for i in range(2):
        match = _make_match(
            db_session, sport, home, away,
            kickoff=now - timedelta(days=1 + i),
            hs=2, as_=0,
        )
        # Predict 0.6 / 0.3 / 0.1 → actual home win → onehot [1,0,0]
        _add_1x2_preds(db_session, match, (0.6, 0.3, 0.1))
    db_session.commit()

    res = compute_live_performance(db_session, market="1x2", window_days=30)
    # Brier per row = (0.6-1)^2 + (0.3)^2 + (0.1)^2 = 0.16 + 0.09 + 0.01 = 0.26
    assert res.n == 2
    assert res.brier is not None
    assert abs(res.brier - 0.26) < 1e-9
    assert res.accuracy == 1.0


def test_skips_match_without_all_selections(db_session: Session) -> None:
    now = datetime.now(UTC)
    sport = m.Sport(code="football", name="Football")
    db_session.add(sport)
    db_session.commit()
    db_session.refresh(sport)
    home = _seed_team(db_session, sport, "HomeFC")
    away = _seed_team(db_session, sport, "AwayFC")

    match = _make_match(
        db_session, sport, home, away,
        kickoff=now - timedelta(days=1),
        hs=1, as_=1,
    )
    # Only "home" prediction stored → incomplete, must be skipped.
    db_session.add(
        m.Prediction(
            match_id=match.id,
            market="1x2",
            selection="home",
            line=None,
            probability=0.5,
            confidence=0.5,
            model_version="v-test",
            drivers={},
        )
    )
    db_session.commit()

    res = compute_live_performance(db_session, market="1x2", window_days=30)
    assert res.n == 0


def test_btts_market_supported(db_session: Session) -> None:
    now = datetime.now(UTC)
    sport = m.Sport(code="football", name="Football")
    db_session.add(sport)
    db_session.commit()
    db_session.refresh(sport)
    home = _seed_team(db_session, sport, "HomeFC")
    away = _seed_team(db_session, sport, "AwayFC")
    match = _make_match(
        db_session, sport, home, away,
        kickoff=now - timedelta(days=1),
        hs=1, as_=1,
    )
    for sel, p in (("yes", 0.7), ("no", 0.3)):
        db_session.add(
            m.Prediction(
                match_id=match.id,
                market="btts",
                selection=sel,
                line=None,
                probability=p,
                confidence=0.6,
                model_version="v-test",
                drivers={},
            )
        )
    db_session.commit()

    res = compute_live_performance(db_session, market="btts", window_days=30)
    assert res.n == 1
    # Brier = (0.7-1)^2 + (0.3-0)^2 = 0.09 + 0.09 = 0.18
    assert res.brier is not None
    assert abs(res.brier - 0.18) < 1e-9
