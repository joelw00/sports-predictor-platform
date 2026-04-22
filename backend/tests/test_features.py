from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.db import models as m
from app.features.football import FootballFeatureBuilder


def _make_match(
    db, *, sport, comp, home, away, kickoff, hs=None, as_=None, status="scheduled"
) -> m.Match:
    match = m.Match(
        sport_id=sport.id,
        competition_id=comp.id,
        home_team_id=home.id,
        away_team_id=away.id,
        kickoff=kickoff,
        status=status,
        home_score=hs,
        away_score=as_,
    )
    db.add(match)
    db.flush()
    return match


def test_elo_updates_on_win(db_session):
    sport = m.Sport(code="football", name="Football")
    db_session.add(sport)
    comp = m.Competition(sport=sport, code="x", name="X")
    db_session.add(comp)
    a = m.Team(name="A", competition=comp)
    b = m.Team(name="B", competition=comp)
    db_session.add_all([a, b])
    db_session.flush()

    t0 = datetime.now(tz=UTC) - timedelta(days=30)
    match1 = _make_match(
        db_session,
        sport=sport,
        comp=comp,
        home=a,
        away=b,
        kickoff=t0,
        hs=3,
        as_=0,
        status="finished",
    )
    match2 = _make_match(
        db_session, sport=sport, comp=comp, home=b, away=a, kickoff=t0 + timedelta(days=7)
    )

    builder = FootballFeatureBuilder()
    builder._update_with_result(db_session, match1)
    feats = builder.snapshot(db_session, match2)
    assert feats.home_elo < feats.away_elo  # A (away) still stronger after beating B
    assert feats.elo_diff < 0


def test_rest_days(db_session):
    sport = m.Sport(code="football", name="Football")
    db_session.add(sport)
    comp = m.Competition(sport=sport, code="x", name="X")
    db_session.add(comp)
    a = m.Team(name="A", competition=comp)
    b = m.Team(name="B", competition=comp)
    db_session.add_all([a, b])
    db_session.flush()

    t0 = datetime.now(tz=UTC) - timedelta(days=30)
    m1 = _make_match(
        db_session,
        sport=sport,
        comp=comp,
        home=a,
        away=b,
        kickoff=t0,
        hs=1,
        as_=1,
        status="finished",
    )
    m2 = _make_match(
        db_session, sport=sport, comp=comp, home=a, away=b, kickoff=t0 + timedelta(days=5)
    )
    builder = FootballFeatureBuilder()
    builder._update_with_result(db_session, m1)
    feats = builder.snapshot(db_session, m2)
    assert 4.5 <= feats.home_rest_days <= 5.5
