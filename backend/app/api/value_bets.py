from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.events import _hydrate_value_bet
from app.db import get_db
from app.db import models as m
from app.schemas import ValueBetOut

router = APIRouter(tags=["value-bets"])


@router.get("/value-bets", response_model=list[ValueBetOut])
def list_value_bets(
    sport: str | None = None,
    market: str | None = None,
    min_edge: float = 0.0,
    min_confidence: float = 0.0,
    day: date | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[ValueBetOut]:
    q = (
        db.query(m.ValueBet)
        .join(m.Match, m.ValueBet.match_id == m.Match.id)
        .filter(m.ValueBet.edge >= min_edge, m.ValueBet.confidence >= min_confidence)
    )
    if market:
        q = q.filter(m.ValueBet.market == market)
    if sport:
        q = q.join(m.Sport, m.Match.sport_id == m.Sport.id).filter(m.Sport.code == sport)
    if day:
        start = datetime.combine(day, time.min, tzinfo=UTC)
        q = q.filter(m.Match.kickoff >= start, m.Match.kickoff < start + timedelta(days=1))
    rows = q.order_by(m.ValueBet.edge.desc()).limit(limit).all()
    return [_hydrate_value_bet(db, vb) for vb in rows]
