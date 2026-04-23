"""Risk management engine.

Given the currently-active :class:`~app.db.models.RiskPolicy`, decides whether
each ranked value bet should actually be placed and how much to stake. Applies:

* a minimum edge and minimum confidence filter (redundant with the value-bet
  engine but declarative on the policy so the UI can tweak it),
* fractional Kelly sizing capped by ``max_stake_pct * bankroll``,
* per-day exposure cap (sum of stakes within the current day),
* maximum number of concurrent positions,
* a drawdown-based stop loss that halts placement once realised drawdown is
  larger than ``stop_loss_drawdown_pct``.

The engine is deliberately stateful-only through its inputs — callers pass the
current bankroll and the list of previously-accepted positions (so it works
identically for live staking and for backtests replaying bet-by-bet).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from sqlalchemy.orm import Session

from app.db import models as m
from app.value_bet.engine import ValueBet


@dataclass
class AcceptedBet:
    kickoff_day: date
    stake: float


@dataclass
class RiskDecision:
    value_bet: ValueBet
    accepted: bool
    recommended_stake: float
    reasons: list[str] = field(default_factory=list)


class RiskGuard:
    """Apply a policy to a sequence of ranked :class:`ValueBet`."""

    def __init__(self, policy: m.RiskPolicy) -> None:
        self.policy = policy

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def evaluate(
        self,
        bets: Sequence[ValueBet],
        *,
        current_bankroll: float,
        starting_bankroll: float | None = None,
        kickoff_lookup: dict[int, datetime] | None = None,
        already_accepted: Sequence[AcceptedBet] | None = None,
    ) -> list[RiskDecision]:
        policy = self.policy
        bankroll = max(0.0, current_bankroll)
        base = starting_bankroll if starting_bankroll is not None else policy.bankroll
        base = max(1e-9, base)
        accepted = list(already_accepted or [])

        decisions: list[RiskDecision] = []
        # Running exposure (by day).
        day_totals: dict[date, float] = {}
        for a in accepted:
            day_totals[a.kickoff_day] = day_totals.get(a.kickoff_day, 0.0) + a.stake

        # Stop-loss short-circuit: if drawdown is already larger than the
        # threshold, every subsequent bet is rejected.
        drawdown = max(0.0, (base - bankroll) / base)
        stop_loss_hit = (
            policy.stop_loss_drawdown_pct > 0
            and drawdown >= policy.stop_loss_drawdown_pct
        )

        for bet in bets:
            reasons: list[str] = []

            if not policy.enabled:
                reasons.append("policy_disabled")

            if bet.edge < policy.min_edge:
                reasons.append(f"below_min_edge({policy.min_edge:.2%})")
            if bet.confidence < policy.min_confidence:
                reasons.append(f"below_min_confidence({policy.min_confidence:.2f})")

            raw_kelly = max(0.0, bet.kelly_fraction) * policy.kelly_fraction
            per_bet_cap = policy.max_stake_pct * bankroll
            stake = min(raw_kelly * bankroll, per_bet_cap)
            stake = max(0.0, stake)

            if stake <= 0:
                reasons.append("kelly_zero")

            # Concurrency cap.
            if (
                policy.max_concurrent_positions
                and len([d for d in decisions if d.accepted]) + len(accepted)
                >= policy.max_concurrent_positions
            ):
                reasons.append(
                    f"max_concurrent_positions({policy.max_concurrent_positions})"
                )

            # Per-day exposure cap.
            kickoff_day: date | None = None
            if kickoff_lookup is not None:
                ko = kickoff_lookup.get(bet.match_id)
                if ko is not None:
                    kickoff_day = ko.date()
            daily_cap = policy.max_daily_exposure_pct * bankroll
            if kickoff_day is not None and daily_cap > 0:
                current_day_total = day_totals.get(kickoff_day, 0.0)
                if current_day_total + stake > daily_cap:
                    reasons.append(f"daily_exposure_exceeded({policy.max_daily_exposure_pct:.0%})")

            # Stop-loss.
            if stop_loss_hit:
                reasons.append(
                    f"stop_loss_drawdown({policy.stop_loss_drawdown_pct:.0%})"
                )

            accepted_bet = not reasons
            if accepted_bet and kickoff_day is not None:
                day_totals[kickoff_day] = day_totals.get(kickoff_day, 0.0) + stake

            decisions.append(
                RiskDecision(
                    value_bet=bet,
                    accepted=accepted_bet,
                    recommended_stake=float(stake),
                    reasons=reasons,
                )
            )

        return decisions


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


DEFAULT_POLICY_NAME = "default"


def get_or_create_default_policy(db: Session) -> m.RiskPolicy:
    """Return the ``"default"`` risk policy, creating it with sane defaults."""
    existing = (
        db.query(m.RiskPolicy).filter(m.RiskPolicy.name == DEFAULT_POLICY_NAME).one_or_none()
    )
    if existing is not None:
        return existing
    policy = m.RiskPolicy(name=DEFAULT_POLICY_NAME)
    db.add(policy)
    db.commit()
    db.refresh(policy)
    return policy
