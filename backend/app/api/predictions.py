from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.db import models as m
from app.schemas import PredictionOut

router = APIRouter(tags=["predictions"])


@router.get("/predictions", response_model=list[PredictionOut])
def list_predictions(
    sport: str | None = None,
    market: str | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[PredictionOut]:
    q = db.query(m.Prediction).join(m.Match, m.Prediction.match_id == m.Match.id)
    if market:
        q = q.filter(m.Prediction.market == market)
    if sport:
        q = q.join(m.Sport, m.Match.sport_id == m.Sport.id).filter(m.Sport.code == sport)
    return [
        PredictionOut.model_validate(p)
        for p in q.order_by(m.Prediction.updated_at.desc()).limit(limit).all()
    ]
