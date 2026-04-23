"""Unit tests for :class:`app.risk.engine.RiskGuard`."""

from __future__ import annotations

from datetime import UTC, datetime

from app.db import models as m
from app.risk.engine import AcceptedBet, RiskGuard, get_or_create_default_policy
from app.value_bet.engine import ValueBet


def _make_policy(
    *,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.5,
    max_stake_pct: float = 0.05,
    max_daily_exposure_pct: float = 0.20,
    max_concurrent_positions: int = 10,
    stop_loss_drawdown_pct: float = 0.20,
    min_edge: float = 0.0,
    min_confidence: float = 0.0,
    enabled: bool = True,
) -> m.RiskPolicy:
    p = m.RiskPolicy(
        name="default",
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        max_stake_pct=max_stake_pct,
        max_daily_exposure_pct=max_daily_exposure_pct,
        max_concurrent_positions=max_concurrent_positions,
        stop_loss_drawdown_pct=stop_loss_drawdown_pct,
        min_edge=min_edge,
        min_confidence=min_confidence,
        enabled=enabled,
    )
    return p


def _bet(
    *,
    match_id: int = 1,
    edge: float = 0.08,
    kelly_fraction: float = 0.10,
    confidence: float = 0.7,
    price: float = 2.0,
) -> ValueBet:
    return ValueBet(
        match_id=match_id,
        market="1x2",
        selection="home",
        line=None,
        bookmaker="Book",
        price=price,
        p_model=0.55,
        p_implied=0.50,
        p_fair=0.50,
        edge=edge,
        expected_value=0.10,
        kelly_fraction=kelly_fraction,
        confidence=confidence,
    )


def test_kelly_sizing_capped_by_max_stake_pct() -> None:
    # Raw Kelly = 0.5 * 0.40 = 0.20 → 200 on bankroll=1000; capped at 0.05 → 50.
    policy = _make_policy(kelly_fraction=0.5, max_stake_pct=0.05, bankroll=1000)
    guard = RiskGuard(policy)
    decisions = guard.evaluate(
        [_bet(kelly_fraction=0.40)],
        current_bankroll=1000.0,
    )
    assert len(decisions) == 1
    d = decisions[0]
    assert d.accepted
    assert d.recommended_stake == 50.0


def test_min_edge_and_min_confidence_filters_reject() -> None:
    policy = _make_policy(min_edge=0.05, min_confidence=0.60)
    guard = RiskGuard(policy)
    decisions = guard.evaluate(
        [
            _bet(edge=0.02),  # below min_edge
            _bet(confidence=0.40),  # below min_confidence
        ],
        current_bankroll=1000.0,
    )
    assert not decisions[0].accepted
    assert any("below_min_edge" in r for r in decisions[0].reasons)
    assert not decisions[1].accepted
    assert any("below_min_confidence" in r for r in decisions[1].reasons)


def test_concurrency_cap_rejects_further_bets() -> None:
    policy = _make_policy(max_concurrent_positions=2)
    guard = RiskGuard(policy)
    bets = [_bet(match_id=i) for i in range(1, 6)]
    decisions = guard.evaluate(bets, current_bankroll=1000.0)
    accepted = [d for d in decisions if d.accepted]
    assert len(accepted) == 2
    # Remaining are rejected with the concurrency reason.
    rest = [d for d in decisions if not d.accepted]
    assert len(rest) == 3
    for d in rest:
        assert any("max_concurrent_positions" in r for r in d.reasons)


def test_daily_exposure_cap_respects_cumulative_stakes() -> None:
    # bankroll=1000, daily cap=10% → 100 total/day; each bet = 50 → 3rd rejected.
    policy = _make_policy(max_daily_exposure_pct=0.10, max_stake_pct=0.05)
    guard = RiskGuard(policy)
    today = datetime(2024, 5, 1, 15, 0, tzinfo=UTC)
    lookup = {1: today, 2: today, 3: today, 4: datetime(2024, 5, 2, 15, 0, tzinfo=UTC)}
    bets = [
        _bet(match_id=1, kelly_fraction=0.40),
        _bet(match_id=2, kelly_fraction=0.40),
        _bet(match_id=3, kelly_fraction=0.40),
        _bet(match_id=4, kelly_fraction=0.40),  # next day — should still fit
    ]
    decisions = guard.evaluate(bets, current_bankroll=1000.0, kickoff_lookup=lookup)
    assert decisions[0].accepted
    assert decisions[1].accepted
    assert not decisions[2].accepted
    assert any("daily_exposure_exceeded" in r for r in decisions[2].reasons)
    assert decisions[3].accepted


def test_stop_loss_short_circuits_all_bets() -> None:
    # Drawdown = 25% > stop_loss 20% → every bet rejected.
    policy = _make_policy(stop_loss_drawdown_pct=0.20, bankroll=1000.0)
    guard = RiskGuard(policy)
    decisions = guard.evaluate(
        [_bet(match_id=i) for i in range(1, 4)],
        current_bankroll=750.0,
        starting_bankroll=1000.0,
    )
    assert all(not d.accepted for d in decisions)
    assert all(
        any("stop_loss_drawdown" in r for r in d.reasons) for d in decisions
    )


def test_policy_disabled_rejects_everything() -> None:
    policy = _make_policy(enabled=False)
    guard = RiskGuard(policy)
    decisions = guard.evaluate([_bet()], current_bankroll=1000.0)
    assert not decisions[0].accepted
    assert "policy_disabled" in decisions[0].reasons


def test_get_or_create_default_policy_is_idempotent(db_session) -> None:
    p1 = get_or_create_default_policy(db_session)
    p2 = get_or_create_default_policy(db_session)
    assert p1.id == p2.id
    assert p1.name == "default"


def test_already_accepted_counts_against_caps() -> None:
    # 2 already-accepted + concurrency cap 3 → only 1 more allowed.
    policy = _make_policy(max_concurrent_positions=3)
    guard = RiskGuard(policy)
    today = datetime(2024, 6, 1, tzinfo=UTC).date()
    accepted = [AcceptedBet(kickoff_day=today, stake=10.0) for _ in range(2)]
    decisions = guard.evaluate(
        [_bet(match_id=i) for i in range(1, 4)],
        current_bankroll=1000.0,
        already_accepted=accepted,
    )
    accepted_count = sum(1 for d in decisions if d.accepted)
    assert accepted_count == 1
