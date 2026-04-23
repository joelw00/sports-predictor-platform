"""API for the risk policy (bankroll / Kelly / exposure limits)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.db import models as m
from app.risk.engine import RiskDecision, RiskGuard, get_or_create_default_policy
from app.schemas import RiskDecisionOut, RiskPolicyOut, RiskPolicyUpdate

router = APIRouter(tags=["risk"])


@router.get("/risk/policy", response_model=RiskPolicyOut)
def get_policy(db: Session = Depends(get_db)) -> RiskPolicyOut:
    policy = get_or_create_default_policy(db)
    return RiskPolicyOut.model_validate(policy)


@router.put("/risk/policy", response_model=RiskPolicyOut)
def update_policy(payload: RiskPolicyUpdate, db: Session = Depends(get_db)) -> RiskPolicyOut:
    policy = get_or_create_default_policy(db)
    for field, value in payload.model_dump().items():
        setattr(policy, field, value)
    db.add(policy)
    db.commit()
    db.refresh(policy)
    return RiskPolicyOut.model_validate(policy)


@router.get("/risk/evaluate", response_model=list[RiskDecisionOut])
def evaluate_current_value_bets(
    sport: str = "football",
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[RiskDecisionOut]:
    """Apply the current risk policy to the most recent value bets in the DB.

    Useful for the operator to see which bets would actually be placed — and
    why others are rejected — before risking real money.
    """
    policy = get_or_create_default_policy(db)
    q = (
        db.query(m.ValueBet)
        .join(m.Match, m.ValueBet.match_id == m.Match.id)
        .join(m.Sport, m.Match.sport_id == m.Sport.id)
        .filter(m.Sport.code == sport)
        .order_by(m.ValueBet.edge.desc())
        .limit(limit)
    )
    rows = q.all()
    if not rows:
        return []

    match_ids = {vb.match_id for vb in rows}
    matches = (
        db.query(m.Match).filter(m.Match.id.in_(match_ids)).all() if match_ids else []
    )
    kickoff_lookup = {mm.id: mm.kickoff for mm in matches}

    from app.value_bet.engine import ValueBet

    bets = [
        ValueBet(
            match_id=vb.match_id,
            market=vb.market,
            selection=vb.selection,
            line=vb.line,
            bookmaker=vb.bookmaker,
            price=vb.price,
            p_model=vb.p_model,
            p_implied=1.0 / vb.price if vb.price > 0 else 0.0,
            p_fair=vb.p_fair,
            edge=vb.edge,
            expected_value=vb.expected_value,
            kelly_fraction=vb.kelly_fraction,
            confidence=vb.confidence,
            rationale=vb.rationale or {},
        )
        for vb in rows
    ]

    guard = RiskGuard(policy)
    decisions = guard.evaluate(
        bets,
        current_bankroll=policy.bankroll,
        starting_bankroll=policy.bankroll,
        kickoff_lookup=kickoff_lookup,
    )

    out: list[RiskDecisionOut] = []
    for d in decisions:
        vb = d.value_bet
        out.append(
            RiskDecisionOut(
                match_id=vb.match_id,
                market=vb.market,
                selection=vb.selection,
                line=vb.line,
                bookmaker=vb.bookmaker,
                edge=vb.edge,
                confidence=vb.confidence,
                kelly_fraction=vb.kelly_fraction,
                recommended_stake=d.recommended_stake,
                accepted=d.accepted,
                reasons=d.reasons,
            )
        )
    return out


# Used by tests and internal calls.
__all__ = ["router", "RiskDecision"]
