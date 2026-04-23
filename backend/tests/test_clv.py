"""Unit tests for CLV (closing line value) tracking in the backtester.

We seed a minimal universe with two snapshots per match (opening + closing)
and verify that:

- ``entry_odds_strategy="closing"`` keeps CLV at 0 and ``n_clv_tracked == 0``
  (no CLV to compute when we're already betting at the close).
- ``entry_odds_strategy="opening"`` computes ``clv = entry / closing - 1`` per
  bet, aggregates ``avg_clv`` / ``clv_win_rate`` / ``n_clv_tracked`` in the
  result, and matches what the hand-computed values should be.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.backtesting.engine import Backtester, _clv
from app.db import models as m
from app.ml.predictor import FootballPredictor
from app.odds.history import closing_price_lookup, entry_snapshots, mark_closing_odds
from app.value_bet.engine import ValueBetEngine


def _seed_two_matches_with_opening_and_closing(db) -> list[m.Match]:
    sport = m.Sport(code="football", name="Football")
    db.add(sport)
    db.flush()
    comp = m.Competition(sport_id=sport.id, code="TCL", name="Test Clv League", country="XX")
    db.add(comp)
    db.flush()
    home = m.Team(competition_id=comp.id, name="Home FC", country="XX")
    away = m.Team(competition_id=comp.id, name="Away FC", country="XX")
    db.add_all([home, away])
    db.flush()
    base = datetime(2025, 1, 1, tzinfo=UTC)
    matches: list[m.Match] = []
    for i, (hs, as_) in enumerate([(2, 1), (0, 2)]):
        match = m.Match(
            sport_id=sport.id,
            competition_id=comp.id,
            home_team_id=home.id,
            away_team_id=away.id,
            kickoff=base + timedelta(days=i),
            status="finished",
            home_score=hs,
            away_score=as_,
            season="2024",
        )
        db.add(match)
        db.flush()
        matches.append(match)
        # Opening snapshot: 3 days pre-kickoff.
        db.add(
            m.OddsSnapshot(
                match_id=match.id,
                bookmaker="snai",
                market="1x2",
                selection="home",
                line=None,
                price=2.50,
                captured_at=match.kickoff - timedelta(days=3),
                is_closing=False,
            )
        )
        # Closing snapshot: 1 minute pre-kickoff. Different price per match.
        closing_price = 2.30 if i == 0 else 2.00
        db.add(
            m.OddsSnapshot(
                match_id=match.id,
                bookmaker="snai",
                market="1x2",
                selection="home",
                line=None,
                price=closing_price,
                captured_at=match.kickoff - timedelta(minutes=1),
                is_closing=False,
            )
        )
    db.commit()
    mark_closing_odds(db, now=base + timedelta(days=365))
    db.commit()
    return matches


def test_clv_formula_positive_and_negative() -> None:
    # Entry 2.50 / closing 2.30 = +8.7% CLV (good).
    assert _clv(2.50, 2.30) > 0
    assert abs(_clv(2.50, 2.30) - (2.5 / 2.3 - 1)) < 1e-9
    # Entry 1.80 / closing 2.00 = -10% CLV (bad).
    assert _clv(1.80, 2.00) < 0
    # Degenerate: closing <= 0 is neutral.
    assert _clv(2.0, 0.0) == 0.0


def test_entry_snapshots_picks_earliest_and_closing_lookup_matches(db_session) -> None:
    matches = _seed_two_matches_with_opening_and_closing(db_session)
    match = matches[0]

    entries = entry_snapshots(db_session, match_id=match.id, market="1x2")
    assert len(entries) == 1
    assert entries[0].price == 2.50  # earliest snapshot

    closing = closing_price_lookup(db_session, match_id=match.id, market="1x2")
    assert closing[("snai", "1x2", "home", None)] == 2.30


def test_backtester_closing_strategy_reports_zero_clv(db_session) -> None:
    _seed_two_matches_with_opening_and_closing(db_session)
    predictor = FootballPredictor()  # fresh (Poisson fallback, no ML artifact)
    bt = Backtester(
        predictor,
        engine=ValueBetEngine(min_edge=0.0, min_confidence=0.0),
        entry_odds_strategy="closing",
    )
    result = bt.run(db_session, market="1x2", label="t-clv-closing")
    assert result.n_clv_tracked == 0
    assert result.avg_clv == 0.0


def test_backtester_opening_strategy_reports_positive_clv(db_session) -> None:
    _seed_two_matches_with_opening_and_closing(db_session)
    # Force a home-win bet on every match by stubbing a predictor that always
    # says P(home) = 0.9 — the 2.50 opening price then has a strong positive
    # edge and the backtester will take it.
    from app.features.football import MatchFeatures
    from app.ml.predictor import MarketProbabilities, PredictionBundle

    class _AlwaysHome:
        version = "stub"

        def predict(self, features: MatchFeatures) -> PredictionBundle:
            return PredictionBundle(
                match_id=features.match_id,
                home_team=features.home_team,
                away_team=features.away_team,
                score_distribution=None,
                markets=[
                    MarketProbabilities("1x2", "home", None, 0.9),
                    MarketProbabilities("1x2", "draw", None, 0.05),
                    MarketProbabilities("1x2", "away", None, 0.05),
                ],
                confidence=0.9,
            )

    bt = Backtester(
        _AlwaysHome(),  # type: ignore[arg-type]
        engine=ValueBetEngine(min_edge=0.0, min_confidence=0.0),
        entry_odds_strategy="opening",
    )
    result = bt.run(db_session, market="1x2", label="t-clv-opening")

    # Both matches offer a home snapshot, and both opened at 2.50. Closings
    # are 2.30 and 2.00, so CLV = (2.5/2.3 - 1) ~ 0.087 and (2.5/2.0 - 1) = 0.25.
    assert result.n_clv_tracked == 2
    assert result.clv_win_rate == 1.0
    expected_avg = ((2.5 / 2.3 - 1) + (2.5 / 2.0 - 1)) / 2
    assert abs(result.avg_clv - round(expected_avg, 5)) < 1e-4
